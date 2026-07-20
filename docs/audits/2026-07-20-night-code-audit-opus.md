# Night code audit — full `src/ai_council/` sweep (Opus)

> **Read-only code audit. No fixes applied. No `src/` file was modified.**

| Field | Value |
|---|---|
| **Model** | Opus 4.8 (1M context), max effort |
| **Session** | Unattended night batch, worktree `night-code-audit`, branch `worktree-night-code-audit` |
| **Audit date** | 2026-07-21 (filename date 2026-07-20 per the operator's explicit path instruction) |
| **HEAD audited** | `f758fa6` — *Merge docs/journal-crux-check-push* |
| **Scope** | Entire `src/ai_council/` tree — 40 files, 7,435 lines — plus `tests/` (28 files, 11,421 lines) and `config/settings.yaml` for config-vs-code cross-checks |
| **Disposition** | Findings only. Nothing struck, nothing filed to `BACKLOG.md` (moratorium). Findings become tickets at a future session on the operator's call. |

## Scope matrix

| Area | Files | Lines | Depth | Method |
|---|---|---|---|---|
| Core orchestration (`cli`, `orchestrator`, `debate`, `synthesis`, `crux_check`, `seat_router`, `output`, `models`, `metrics`, `runner`, `policy`, `routing`, `inbox`) | 13 | ~4,300 | Full read | Direct (this session) |
| `providers/` + `healthcheck`, `mode_detector`, `doctor` | 11 | ~1,700 | Full read | Delegated sub-audit, spot-verified against source |
| `research/` subtree | 12 | ~1,400 | Full read | Delegated sub-audit, spot-verified against source |
| `tests/` | 28 | 11,421 | Full read of the 8 largest, skim + targeted read of the rest | Delegated sub-audit |
| `config/settings.yaml` ↔ code | — | — | Key-by-key cross-check | Direct + delegated |

**Verification note.** Every P1 below was confirmed by reading the cited source in this session; four were additionally confirmed by *executing* the logic (marked **[executed]**). Claims that survived only as inference are marked **[inferred]** and stated as such. Complexity numbers are `ruff` output, not estimates.

---

# P1 — will bite

## Correctness

### P1-1 · `_parse()` runs outside `generate()`'s exception guard
`src/ai_council/providers/base.py:231`

The `try` at `:219-229` wraps only `_invoke`. `parsed = self._parse(raw)` sits *after* it, so any `AttributeError` / `IndexError` / `ValueError` from a malformed SDK payload escapes **unwrapped** — e.g. `parse_openai_chat` at `:157` does `raw.choices[0]` with no guard on the attribute existing.

**Why it bites:** it breaks the `Raises: ProviderError` contract documented at `:214-216`, and that contract is load-bearing downstream. `synthesis.py:109` and `crux_check.py:214` both catch **only** `ProviderError`. Worse, `CruxCheckService.check`'s docstring at `crux_check.py:203` reads *"Never raises."* — a claim the code cannot honour. The debate survives today only because `debate.py:244` has a blanket `except Exception`; that is defence in depth accidentally covering a contract hole, not a design.

**Fix direction:** move `_parse` inside the `try`, or give it its own `try` → `ProviderError`.

### P1-2 · One seat's non-`ProviderError` kills the entire debate round
`src/ai_council/debate.py:279` · `src/ai_council/seat_router.py:84` · `src/ai_council/providers/cli_base.py:267,330`

`asyncio.gather(*tasks)` is called **without `return_exceptions=True`**. `SeatRouter.try_cli` catches only `ProviderError` (`seat_router.py:84`), and `_run_seat` (`debate.py:265`) adds no guard. Reachable raisers in the CLI lane: `doc.get("result")` at `cli_base.py:267` when the CLI emits a JSON array (`AttributeError`; only `JSONDecodeError` is caught at `:265`), and `int(...)` at `cli_base.py:330` (`ValueError` on empty string).

**Why it bites:** the failure mode is not "one seat degrades" but "the round raises, every sibling seat is cancelled, no API fallback fires, and no `fallback_event` is recorded" — the exact opposite of the same-seat-fallback guarantee `seat_router.py`'s module docstring advertises at `:1-18`.

**Fix direction:** `return_exceptions=True` on the gather plus an isinstance triage, and wrap `_extract`/`communicate` failures in `ProviderError` at the raise site.

### P1-3 · SDK clients are built in `__init__` and then reused across multiple event loops
`src/ai_council/providers/base.py:183` → `openai_provider.py:17`, `anthropic.py:17`, `xai.py:19`, `deepseek.py:19` · driven by `src/ai_council/cli.py:635` + `cli.py:764`

`AIProvider.__init__` calls `self._configure()` at `base.py:183`, and four of the five providers construct their `AsyncOpenAI` / `AsyncAnthropic` client there — outside any running loop. `cli.py:635` builds the provider pool **once**, then the inbox batch loop calls `asyncio.run(runner.run(...))` **per file** at `cli.py:764`. Each `asyncio.run` creates and destroys a fresh loop while the same client objects — and their `httpx` connection pools, bound to the *first* loop — are reused.

**Why it bites:** this is precisely the failure class the repo already documented for one provider. `CLAUDE.md` §10 records *"google-genai: `genai.Client(api_key=...)` must be created INSIDE the async method, NOT in `__init__`"*, and `gemini.py:23` correctly complies with a docstring explaining the event-loop rationale. **The other four providers violate the same rule and nobody noticed, because the single-question path only ever runs one loop.** A `--inbox` batch with ≥2 files is the trigger.

**Fix direction:** build the client lazily inside `_invoke` (as `gemini.py` does), or hold one loop for the whole batch instead of one per file.

### P1-4 · `--file` silently ignores frontmatter `models:`, and `--full` is not the documented no-op **[executed]**
`src/ai_council/cli.py:687` (inbox) vs `cli.py:811-814` (interactive) · help text at `cli.py:487`

The two paths compute the same precedence with different expressions, and they disagree. Executed truth table for a file whose frontmatter says `models: claude,gemini`:

| flags | interactive (`--file`) | inbox (`--inbox`) |
|---|---|---|
| *(default)* | **`None` — ignored** | `claude,gemini` — honored |
| `--full` | `None` | `None` |
| `--lite` | `claude,gemini` | `claude,gemini` |
| `--lite --full` | `None` | `None` |

Two independent defects sit in that table:

1. **`council --file q.md` silently discards the file's `models:` panel override.** `eff_full = (use_full_panel or not lite) or ...` is `True` by default (because `not lite` is `True`), and the frontmatter read is gated on `not eff_full`. The operator gets the default panel and no warning. The same file through `--inbox` gets the panel it asked for.
2. **`--full` is documented as `"[No-op] Full panel is now the default"` (`cli.py:487`) but materially changes behaviour** — it is read at `:687`, `:688`, `:811` and flips whether frontmatter `models:` is honored in the inbox path.

**Why it bites:** this is the repo's own known blind spot — `CLAUDE.md` §10 *"Inbox loop parity: Features added to interactive CLI must be explicitly mirrored into inbox loop"* — no longer hypothetical but live, and pointing the *opposite* way from the documented direction (here the inbox path is the correct one). Silent wrong-panel selection is unfalsifiable from the transcript, which records the panel that ran, not the one requested.

**Fix direction:** extract one `resolve_run_overrides(flags, meta, config)` helper and call it from both paths; correct or remove the `--full` help text.

### P1-5 · The "best-effort" secondary mirror is unguarded and can abort the run
`src/ai_council/output.py:268` vs its own contract at `output.py:171,255-256`

`_write_routed` wraps the `target_paths` mirror writes in `try/except` (`:306-313`) but leaves the `secondary_dir` write bare:

```python
if secondary_dir.exists():
    secondary_path = secondary_dir / filename
    secondary_path.write_text(content, encoding="utf-8")   # ← unguarded
```

**Why it bites:** the docstrings state *"Optional mirrors stay best-effort"* (`:171`) and *"Best-effort destinations (`secondary_dir`, `target_paths`) are never reported this way — they only warn"* (`:255-256`). The code does not honour that for `secondary_dir`. A `PermissionError` — routine on Windows when the file is open in another program — propagates out of `_write_routed`, out of `save_to_file`, and aborts `CouncilRunner.run` **after** the canonical transcript landed but **before** the minority report, the verdict package, and every `--return-dir` write. A legacy mirror takes down the required deliverables, and the R4 accumulate-then-raise design at `orchestrator.py:197-201` — built specifically to prevent this shape of loss — is bypassed entirely because the exception is an `OSError`, not a `RoutingFailure`.

**Fix direction:** wrap it in the same `try/except` as `target_paths`.

### P1-6 · Degraded research reports are written to a 7-day cache
`src/ai_council/research/runner.py:217` · `research/cache.py:114` · `config/settings.yaml:439`

`cache_put` is called unconditionally when `not no_cache`, with no `report.degraded` guard. `cache.py:114` persists the flag and `:94` restores it.

**Why it bites:** one transient outage that drops the panel below `min_successful_providers` freezes DEGRADED into the cache for `cache_ttl_days: 7`. Every subsequent run of that query returns the cached degraded report and exits 3 (`cli.py:876-877`) **without ever re-calling a provider** — so the condition cannot self-heal and looks, to the operator, like a persistent provider outage.

**Fix direction:** skip `cache_put` when `report.degraded`.

### P1-7 · `classify_error` substring-matches naked HTTP digits, and `auth` outranks `server_error`
`src/ai_council/providers/base.py:65,68-77,86`

The classifier lowercases `str(exc)` and scans for `"429"`, `"401"`, `"403"`, `"404"`, `"500"`, `"502"`, `"503"` anywhere in the message, plus the bare token `"auth"`. Two consequences, both demonstrated against the real function:

- Request IDs, model names and token counts routinely carry those digits — `classify_error("model gpt-4290 unavailable")` returns `rate_limit`; `classify_error("request req_5031 failed: content_policy violation")` returns `server_error` (retryable) because the `"503"` arm at `:86` fires before the content-policy arm at `:88`.
- The `auth` arm at `:72` is checked **before** `server_error` at `:86`, so `"500 internal error: authentication service degraded"` classifies as `auth` — **non-retryable**. `debate.py:68` then breaks the retry loop and burns the seat on a fully recoverable failure.

**Why it bites:** misclassification is silent in both directions — a retryable failure is abandoned, and a permanent failure is retried into the same wall. Neither leaves a distinguishable trace.

**Fix direction:** classify off the SDK exception type / `exc.status_code`, not the rendered string.

### P1-8 · `provider_statuses` means "ever succeeded", but the contract surface reads it as "status"
`src/ai_council/debate.py:221,285` → `src/ai_council/output.py:1160,1251`

`debate.py:221` seeds every provider to `"failed"`; `:285` flips a provider to `"ok"` on success **in any round** and never flips it back. `output.py:1160` then derives `dropped = [k for k, v in provider_statuses.items() if v == "failed"]` and feeds it to the verdict package's `panel.dropped` and `degradation.failed_providers` (`:1251`).

**Why it bites:** a provider that answers Round 1 and then dies in Round 2 is recorded `"ok"`, appears in `seated`, is absent from `dropped`, and — because `outcome.degraded` is only set on the total-failure path (`debate.py:288-304`) — produces `degradation.degraded == false`. **Mid-debate seat loss is invisible in every field of the contract-version-1.0 delegation surface.** The two-signal rule the payload comment claims at `output.py:1222-1224` does not fire for the most common partial-failure shape.

**Fix direction:** track per-round participation and derive `dropped` from "did not respond in the final round", or add an explicit `partial` status.

### P1-9 · A synthesis failure discards the entire run, including everything already paid for
`src/ai_council/synthesis.py:116,119`

`synthesize` re-raises `ProviderError` at `:116` and raises a bare `RuntimeError` on empty content at `:119`. Neither is caught in `orchestrator.py`, so `CouncilRunner.run` unwinds before reaching *any* writer at `:204-265`.

**Why it bites:** by that point the run has paid for every panelist across every round. A synthesizer hiccup — the single most replaceable call in the pipeline — destroys the transcript, the verdict package, and the metrics sidecar. This directly contradicts the degradation philosophy the rest of the codebase is built on: `debate.py:291` returns partial rounds rather than aborting, `output.py:530-548` deliberately downgrades a metrics-sidecar failure to a degradation note *specifically so* the caller's deliverables survive, and `crux_check.py` never raises by design.

**Fix direction:** catch at the orchestrator, emit the transcript with a `**Status:** SYNTHESIS FAILED` header and a degradation entry, and exit non-zero — the debate content is the expensive artifact and it is already in hand.

### P1-10 · The research summarizer is the one network call with no timeout and no retry cap
`src/ai_council/research/merger.py:170-175`

`AsyncOpenAI(**client_kwargs)` is built with only `api_key` (+ optional `base_url`), then `await client.chat.completions.create(...)` runs with no `asyncio.wait_for`. SDK defaults apply: 600 s timeout × `max_retries=2` → **up to ~30 minutes of silent hang**, occurring *after* the live display has shown every provider as done. Every retrieval provider passes an explicit timeout (`perplexity.py:56`, `openai_mini_research.py:60`, `openai_deep_research.py:59`); the summarizer model's configured `timeout_sec` (`settings.yaml:97`) sits unread on `model_cfg`.

**Fix direction:** `timeout=model_cfg.timeout_sec, max_retries=1`.

### P1-11 · Research timeouts are detected by substring-sniffing an error message that never contains the substring
`src/ai_council/research/display.py:119,176`

`statuses[name] = "timeout" if "Timed out" in str(exc) else "error"`. But an SDK-level timeout raises `APITimeoutError`, which the providers wrap as `f"API timeout: {exc}"` (`perplexity.py:75`, `grok_research.py:78`, `openai_mini_research.py:73`, `openai_deep_research.py:72`) — a string that does **not** contain `"Timed out"`. Gemini has no timeout branch at all and folds everything into `"API error: {exc}"` (`gemini_research.py:67-68`).

**Why it bites:** real timeouts are misclassified as generic errors in the live table, in `ResearchResult.timed_out`, and therefore in the saved report (`research/output.py:92`) — so the one signal that tells the operator "raise the timeout" never appears.

**Fix direction:** a `ResearchProviderTimeout(ResearchProviderError)` subclass classified by `isinstance`.

## Architecture

### P1-12 · `cli.run()` is a 196-statement, 56-branch, complexity-50 function
`src/ai_council/cli.py:551`

`ruff` output, unedited:

```
cli.py:551:5: C901    `run` is too complex (50 > 10)
cli.py:551:5: PLR0913 Too many arguments in function definition (20 > 5)
cli.py:551:5: PLR0912 Too many branches (56 > 12)
cli.py:551:5: PLR0915 Too many statements (196 > 50)
```

It performs, in one body: Windows encoding setup, dotenv loading, config loading, empty-key stripping and *reloading*, output-dir resolution, return-dir resolution, target resolution, mode validation, provider construction, health-check gating, policy construction, a ~130-line inbox batch loop with its own precedence logic and exit-code accumulator, and a ~120-line single-question path with mode auto-detection, research dispatch, and debate dispatch.

**Why it bites:** this is the direct structural cause of **P1-4**. Two near-identical precedence blocks 120 lines apart, in a function no reviewer can hold in their head, is exactly the machine that produces the inbox-parity divergences `CLAUDE.md` §10 warns about. It is also why `effective_synthesizer` is computed at `:609` and then dead — silently overwritten at `:806` before its only read at `:895`.

**Fix direction:** extract `_resolve_overrides()`, `_run_inbox_batch()`, and `_run_single()`; keep `run()` as flag-parsing plus a three-way dispatch.

### P1-13 · `output.py` is four modules wearing one name
`src/ai_council/output.py` (1,352 lines — 18% of `src/`)

It contains, with no internal seam:

| Responsibility | Lines | Belongs in |
|---|---|---|
| Rich console presentation | `65-143` | presentation |
| File I/O + routing policy + the `RoutingFailure`/`OutputRoutingError` contract | `146-322`, `466-555`, `642-708`, `1258-1352` | infrastructure |
| **A hand-rolled CommonMark inline parser** | `771-1039` | a library (see buy-vs-build) |
| Domain heuristics + the contract-v1.0 verdict schema | `558-639`, `1066-1255` | domain |

**Why it bites:** three concrete symptoms already in the tree. (a) `research/output.py:10` reaches across the package to import the **private** `_write_routed`, because the only way to reuse the routing policy is to import from the presentation module. (b) The module-level `console = Console(...)` at `:28` is a singleton that `orchestrator.py:54` and `cli.py:46` each duplicate with *their own* `Console` instances — three writers to one stdout, which is why `--json` output is unparseable (**P2-6**). (c) The verdict-package schema — the inter-repo delegation contract — is defined inside a module whose docstring says *"Rich console output and markdown file save"*.

**Fix direction:** split into `presentation/console.py`, `io/routing.py`, `domain/verdict.py`; replace the parser with a library. `_write_routed` becomes public `write_routed` at that point.

## Test integrity

### P1-14 · A test named `test_skips_failed_results` is a literal tautology
`tests/test_research.py:154`

```python
assert "fail" not in report.merged_report.upper() or "fail" in report.merged_report.lower()
```

The first clause uppercases the haystack and searches for a lowercase needle — unconditionally true (confirmed by execution against the string `"fail everywhere FAIL"`). `merge_results` could splice the errored provider's body verbatim into the report and this passes.

### P1-15 · The ADR-03 blind-voting test asserts nothing about anonymization
`tests/test_debate.py:151-223`

The capturing wrapper installed at `:189` is unconditionally overwritten at `:194-213` before `run_debate` is ever called. `captured_prompts` (`:155`) stays empty and is never read. The only surviving assertions are `len(outcome.rounds) == 2` and `outcome.rounds[1].number == 2`.

**Why it bites:** ADR-03 blind voting is the one invariant `CLAUDE.md` §10 flags as un-changeable without an ADR, and a Round-2 prompt leaking every provider name passes this test.

### P1-16 · The two orchestration functions in the codebase are mocked out of existence in every test that names them
`src/ai_council/research/runner.py:134` · `src/ai_council/orchestrator.py:39`

- `run_research` is patched at `tests/test_research.py:1552,1671,1719,1752` and **never invoked**. Its cache-hit branch, provider fan-out, degradation-threshold wiring and routing — 238 lines — are unexercised.
- `tests/test_runner.py:209-230` patches `run_debate`, `synthesize`, `save_to_file` and all three print functions, then asserts `result is fake_result` — which is exactly what the `synthesize` mock returns. It passes against a `run()` with the entire panel-selection, seat-routing, crux and output pipeline deleted.
- The orchestrator's R4 accumulate-then-raise sequencing (`orchestrator.py:197-279`) has **zero** coverage at the site that implements it. `tests/test_output.py:143-178` validates the contract by hand-replicating the call sequence *inside the test*, proving the writers work but not that the orchestrator wires them that way.

**Why it bites:** the units are well proven; the code that *sequences* them is not. P1-5 and P1-9 both live in exactly that gap.

---

# P2 — real but bounded

**Correctness / behaviour**

- **P2-1 · The crux extraction call renders as "Round -1" in the operator's cost summary [executed].** `crux_check.py:220` bills the call with `round_number=-1`; `output.py:115-121` routes any non-zero round into `by_round` and `sorted()` puts `-1` first. Verified output: `+-- Round -1: $0.0012 (1 providers, 150 tokens)`. The same `-1` is written to the `_metrics.json` sidecar (`output.py:1319`).
- **P2-2 · Three incompatible conventions for `round_number`.** Real rounds are `1..N`; synthesis is `len(rounds)+1` on the `ModelResponse` (`synthesis.py:107`) but `0` in its metrics (`:137`); crux is `-1`. One field, three meanings, no documented mapping.
- **P2-3 · Transcript filenames can silently overwrite.** `output.py:491` builds `council-out-{_ts()}-{mode}-{slug}.md` at 1-second resolution with no collision check; `write_text` truncates. Two inbox files with the same cleaned slug processed within one second lose the first.
- **P2-4 · The verdict package advertises its own path before writing it.** `output.py:1206-1212` appends the `verdict` artifact entry with `guaranteed_dirs` paths, bypassing the `p.exists()` filter applied to every other kind at `:1194-1204` — whose comment claims the filter makes a phantom path *"structurally impossible regardless of who populates `written`"*. If the `--return-dir` write then fails, the canonical package names a return-dir path that does not exist.
- **P2-5 · `pick_synthesizer` silently ignores `--synthesizer` and then labels the result "(user-selected)".** `runner.py:72` falls through to `all_providers[not_in_panel[0]]` — dict order — when the preferred synthesizer is in the panel; `orchestrator.py:101` still prints `(user-selected)` because `synthesizer_specified` is `True`.
- **P2-6 · `--json` output is unparseable.** `research/runner.py:182,235` dump JSON to stdout *after* `print_research_summary` wrote the human summary to the same stream (`cli.py:46` builds the console without `stderr=True`). `ai-council research "q" --json | jq` cannot work.
- **P2-7 · `SeatMetrics` scalar fields are last-write-wins across rounds.** `seat_router.py:104-111`: a seat that runs CLI in rounds 1–2 and falls back in round 3 ends with `actual_backend="api"` while `actual_model` still holds the **CLI** model string — an internally inconsistent telemetry record. Only `fallback_events` preserves the history.
- **P2-8 · Grok research never received the SDK timeout/retry fix.** `research/providers/grok_research.py:54` builds `AsyncOpenAI` with neither `timeout=` nor `max_retries=`, while its three siblings pass both with comments calling it "Fix-A parity".
- **P2-9 · A Gemini research timeout abandons a live billable background job.** `gemini_research.py:57` wraps a `background=True` interaction (`:85`) in `asyncio.wait_for` with no cancel on the way out.
- **P2-10 · `warnings.catch_warnings()` held across an `await` in a concurrent fan-out.** `gemini_research.py:78-86` — `catch_warnings` mutates process-global state and is not safe across a suspension point with up to five providers on one loop.
- **P2-11 · No `AsyncOpenAI` client in `research/` is ever closed, and a new one is built per call.** `perplexity.py:53`, `grok_research.py:54`, `openai_mini_research.py:58`, `openai_deep_research.py:57`, `merger.py:174` — zero `close()`/`aclose`/`async with` in the subtree. `gemini.py:23` leaks one per call for the same reason.
- **P2-12 · `_split_sections` has no fenced-code awareness.** `output.py:592` treats any line starting `## ` as a heading, including inside a ``` fence — so a synthesis containing a shell snippet fabricates a section. Inconsistent with the CommonMark rigour 200 lines below it.
- **P2-13 · `healthcheck` re-implements a timeout the base already exposes and reaches into `_config` to do it.** `healthcheck.py:34-37,44-47` uses `getattr(provider, "_config", None)` though `AIProvider.timeout_sec` (`base.py:193`) exists for exactly this, and wraps `generate()` in an outer `wait_for` instead of passing its `timeout=` parameter — so the inner guard still runs at 240 s while the outer caps at 60.
- **P2-14 · `mode_detector`'s hardcoded "cheapest" order contradicts the configured costs.** `mode_detector.py:12` ranks `openai` (2.50/1M) ahead of `grok` (2.00/1M) — `settings.yaml:61,89` has the data to compute it.
- **P2-15 · The ADR-08 degradation alarm defaults to off.** `research/merger.py:91` — `min_successful: int | None = None`, and `:117-119` makes `degraded` unconditionally `False` when it is `None`. A caller that forgets the kwarg gets a silently non-degraded report and exit 0.
- **P2-16 · Cache round-trip drops `sources`, producing two contradictory counts in one summary.** `research/cache.py:74-83` omits `sources=`; `research/output.py:144-145` then prints "0 sources total" two lines above `:163`'s real count from meta.
- **P2-17 · In-flight provider tasks are never cancelled on abnormal exit.** `research/display.py:130-160` — `_run_one` catches `Exception` but not `BaseException`, so `KeyboardInterrupt` orphans the pending tasks and their HTTP requests.
- **P2-18 · Temp-dir cleanup can mask the real error.** `cli_base.py:176,200-204` — the child's `cwd` *is* the scratch dir and the `finally` kills the tree without awaiting, so `TemporaryDirectory.__exit__` can raise `PermissionError` over an in-flight `ProviderError`.

**Documentation ↔ behaviour divergence**

- **P2-19 · `--synthesizer` help says "Defaults to gemini".** `cli.py:516`. `config/settings.yaml:12` is `synthesizer: "openai"` (ADR-01 revised 2026-07-18). The operator-facing string still names the pre-ruling default.
- **P2-20 · `research/headless.py` docstring claims it "touches none of that machinery".** `headless.py:1` vs `:25` — it imports `_error_result` from `display.py`, dragging the whole Rich display stack into the headless path at import time.
- **P2-21 · `merger.py` docstring promises source dedup the merged document does not have.** `merger.py:4` — `_deduplicate_sources` (`:57`) is consumed only for its `len()`; the deduplicated list is discarded and `_build_merged_document:79-81` re-emits each provider's raw list, so URLs genuinely repeat.
- **P2-22 · `summarize_report` docstring says "cheapest available model".** `merger.py:140,148` reads one hardcoded config name with no cost comparison.
- **P2-23 · `gemini_research.py:7` states a config fact that is false.** Docstring names `deep-research-pro-preview-12-2025` "(configured in settings.yaml)"; `settings.yaml:485` configures `deep-research-preview-04-2026`. The class default at `:33` carries the same stale ID.
- **P2-24 · `_print_research_paths` does not do what its docstring says.** `research/runner.py:123-131` labels a **required** `--return-dir` deliverable "Copied:", and its missing-secondary warning sits in an `elif` that is skipped whenever any other extra path was written.
- **P2-25 · `summary_2500` holds the full merged document on the headless path.** `merger.py:125` sets it as a placeholder and `headless.py:81-87` never calls `summarize_report`, while `research/models.py:34` documents it as a "2.5K token summary".
- **P2-26 · `CLI_FALLBACK_CAUSES` / `NON_RETRYABLE_ERRORS` are declared but never read.** `base.py:28,39` — the comment at `:32-38` claims the frozenset is *"one source, not two lists to drift"*; grep across `src/` and `tests/` finds zero consumers. The anti-drift guarantee is documentation-only.

**Config ↔ code**

- **P2-27 · Dead config keys `sdk` and `token_budget`.** `settings.yaml:45,55,65,74,83,93` and `:159,169,215,225`, parsed into typed dataclasses at `config_loader.py:19,74`, **zero reads in `src/`** (verified by grep). `sdk: "openai-compatible"` reads as if it drives provider dispatch; dispatch is actually the hardcoded map at `cli.py:48`.
- **P2-28 · Dead config key `research.providers.perplexity.base_url`.** `settings.yaml:452` → `config_loader.py:425`, but `research/runner.py:71-79` never passes it and `PerplexityProvider.__init__` has no such parameter — the URL is hardcoded at `perplexity.py:14`. (Grok's *is* threaded, at `runner.py:114`.) Same defect class the crux-check block explicitly guards against at `settings.yaml:406-409`.
- **P2-29 · Provider constructor defaults have drifted from config.** `perplexity.py:29` defaults `timeout_sec=60` against `settings.yaml:455`'s `240`, whose comment records that 60 was *measured* too tight; `grok_research.py:33` defaults `120` against `:480`'s `300`. The x.ai base URL is written in three places.
- **P2-30 · Gemini research cost is structurally always $0.** `settings.yaml:489-490` sets both rates to `0.0`, so `gemini_research.py:119-122` always computes zero and `merger.py:109` sums a total that omits the panel's most expensive provider (1800 s deep research) — against ADR-06's per-provider tracking mandate.
- **P2-31 · Hardcoded prompts and timeouts against `CLAUDE.md` §5.8.** `mode_detector.py:14-22,29`, `healthcheck.py:22-24`, `cli_base.py:150`, `doctor.py:275`.
- **P2-32 · The entire `cli_base.py` subtree is unreachable from shipped config.** No model in `settings.yaml:43-100` declares `backend`/`cli_command`, so `cfg.backend == "cli"` at `seat_router.py:136` is never true — 337 lines of security-sensitive subprocess code with no production path (and its blocking `subprocess.run` at `:94-97,148-152` is latent for the same reason).

**Test quality**

- **P2-33 · `merger.summarize_report` has no test.** `research/merger.py:135` — only its `_truncation_fallback` escape hatch is covered (`tests/test_research.py:212-226`), so a summarizer that always silently degrades to truncation is indistinguishable from a working one.
- **P2-34 · The crux-check operator signal is untested.** `orchestrator.py:175-187` — three console branches, and `tests/test_debate.py:757-759` explicitly reasons that this console line is the *sole* operator-facing indication of retrieval failure. No test asserts it prints.
- **P2-35 · Weak-disjunction assertions that invert their own guard.** `tests/test_research.py:1520` (`"WARNING" not in text.upper() or "Degraded" not in text` — a banner reading `WARNING: degraded` passes) and `tests/test_debate.py:45` (`or` where `and` is meant — dropping half the panel passes).
- **P2-36 · Structural proxies masquerading as behavioural guarantees.** `tests/test_research_headless.py:147-159` asserts `not hasattr(headless_mod, "cache_get")`; switching to `from ... import cache` + `cache.cache_get(...)` passes while reading the cache the docstring forbids.
- **P2-37 · Assertions too weak to fail.** `tests/test_research.py:1462` (`assert "2" in out` — a bare digit in a summary full of counts and costs); `tests/test_cli.py:563,571` (`exit_code != 2` passes on a silent exit 0); `tests/test_cli.py:574-579` (`--modes` asserts only exit 0, never that modes printed).
- **P2-38 · `create=True` patches that fabricate an attribute nothing reads.** `tests/test_research.py:1553,1672,1720,1753` — `ai_council.cli` has no module-level `run_research` (it is imported inside `_run_research_dispatch`, `cli.py:102`).
- **P2-39 · `tests/test_integration.py` verifies "did not crash" and no-ops silently without keys.** Assertions at `:87,111` are `content` truthy / `latency_sec > 0` / `len > 500`; the module-level skip at `:33` means keyless CI reports green rather than "not run".
- **P2-40 · Cross-test coupling on the real system temp dir.** `tests/test_cli.py:970-973,985` and `tests/test_doctor.py:463-468` census `%TEMP%` for `aicouncil-scratch-*`; the `finally` cleanups at `tests/test_cli.py:1064-1068,1100-1104` exist because the coupling is real.

---

# P3 — nits

**Dead code / dead values**
- `cli.py:609` — `effective_synthesizer` assigned, then overwritten at `:806` before its only read. Dead.
- `output.py:49` — `import re` inside `_slug` shadowing the module-level import at `:5`.
- `research/display.py:135` — `done_tasks` declared, unioned at `:149`, never read.
- `research/models.py:12` — `Source.snippet` never assigned or read anywhere in `src/`.
- `research/provider.py:36` — `model_string()` abstract, implemented by all five, called nowhere in `src/`.
- `research/cache.py:133` — `cache_invalidate` has no production caller; no CLI flag reaches it.
- `crux_check.py:276,287` — `attempted or len(report.results)`; `build_crux_check_service:307` guarantees `attempted >= 1`, so the `or` arm is unreachable.
- Five dead module loggers: `anthropic.py:10`, `gemini.py:11`, `openai_provider.py:10`, `xai.py:10`, `deepseek.py:10` — assigned, zero `logger.` calls.
- `cli_base.py:72` — `ALL_PROXY` is in `_PROXY_VARS` but not `_ENV_ALLOWLIST`, so the userinfo-strip loop at `:81-83` can never see it.

**Naming / boundaries**
- `_Parsed` (`base.py:136`) is underscore-private yet imported by five providers, `tests/conftest.py:19`, `tests/test_debate.py:16`, and is the declared return type of a public abstract method at `:266`. Same class of smell as `_write_routed` and `_error_result`. Rename all three.
- Vocabulary drift — provider / seat / backend / model / panelist name the same entity depending on the file (`base.py:186` vs `:194`, `cli.py:48` `PROVIDER_CLASSES` vs `seat_router.py:32` `CLI_PROVIDER_CLASSES`). No glossary.
- Two different `_run_one` contracts under one name: `research/headless.py:35` returns `ResearchResult`, `research/display.py:108` returns `tuple[str, ResearchResult | None]` — and `headless.py` imports from `display.py`, so both are in scope for a reader.
- `research/merger.py:110` — `total_duration_sec` is a `max`, not a total. Correct semantics for a parallel fan-out, wrong word.
- `research/display.py:35` — the five statuses are a stringly-typed state machine documented in a trailing comment; `StrEnum`/`Literal` is the 3.12 idiom and would make `:119` type-checkable.
- `seat_router.py:81` — `seat.cli = {"name": ..., "version": ...}` is a raw dict against the repo's own "dataclasses, not raw dicts" standard (`models.py:65` types it `dict | None`).

**Type honesty**
- `debate.py:90` — `assert last is not None` in production code; under `python -O` the assert is stripped and the function returns `None` against its declared `ModelResponse | ProviderError`.
- `research/display.py:165` — `# type: ignore[misc]` hides a genuine `ResearchResult | None` returned as `list[ResearchResult]`, plus a key collision when two providers share a `name()`.
- Three `# type: ignore` on annotation iteration (`grok_research.py:154`, `openai_mini_research.py:156`, `openai_deep_research.py:157`) whose comment blames the SDK; the parameter is annotated `object` *by this code* at `grok_research.py:149`. Self-inflicted.
- `research/runner.py:69` — `_instantiate_provider(name, p_cfg, api_key)` with `p_cfg` unannotated (implicit `Any`), so none of the five differing kwarg shapes type-check.
- `runner.py:15` — `provider_classes: dict`; `cli.py:787` — `file_meta: dict`. Bare `dict` against "type hints everywhere".
- `cli_base.py:242` — `_parse` returns `Any` instead of `_Parsed`, weakening the ABC signature.
- 16 × `ANN401` (`Any`) across the provider `_invoke`/`_parse` boundary — defensible as raw-SDK types, but undeclared as a deliberate exception.
- 3 × `B905` `zip()` without `strict=` at `debate.py:40,42,282` — the anonymization label/response pairing silently truncates on a length mismatch.
- 3 × `B023` loop-variable capture at `debate.py:268,270,273` — safe today only because `gather` is awaited inside the same iteration.

**Duplication**
- `_extract_content`/`_extract_sources`/`_collect_annotations` triplicated near-verbatim: `openai_mini_research.py:110-161`, `openai_deep_research.py:113-162`, `grok_research.py:107-159` (~50 lines × 3, differing only in a skipped item-type tuple).
- `xai.py:16-29` and `deepseek.py:16-29` are near-byte-identical outside class names. *Per repo rule the classes stay separate* — but `_configure`/`_invoke` could move to shared helpers beside the existing `parse_openai_chat` (`base.py:150`) without merging anything.
- `research/runner.py:169-184` vs `:221-237` — the cache-hit and fresh emission blocks are copy-paste twins, including duplicated function-local imports.
- `research/runner.py:41-42` vs `:190-191` — panel-filter logic computed twice; change one and the ADR-08 degradation denominator drifts from the panel.
- `_format_duration` duplicated byte-for-byte: `research/display.py:25-30` and `research/output.py:28-33`.
- Degradation arithmetic duplicated at `research/output.py:78-79` and `:168-169`.

**Small correctness**
- `research/merger.py:75` — `hasattr(result, "model")` is always `False` (`ResearchResult` has no `model` field), so every heading renders `## Report from GROK (grok)`.
- `research/display.py:127` — a bare `raise SomeError()` yields `str(exc) == ""`, which is falsy at `research/output.py:92,142`, so a failed provider renders as status "ok" and is excluded from the "N failed" count.
- `research/output.py:21-25` — `_slug` can return empty on a punctuation-only query, yielding `council-out-<ts>-research-.md`.
- `research/providers/gemini_research.py:92` — sleeps `poll_interval_sec` *before* the first status check, costing a fixed 10 s even on instant completion.
- `cli_base.py:153` — `_read_version` raises `IndexError` on whitespace-only stdout, swallowed by the blanket `except Exception` at `:154` and misreported as "no version".
- `healthcheck.py:53-54` — `_HEALTHCHECK_MESSAGES` replaces the real error text, so "model not found (check model string in settings.yaml)" never says *which* model.
- `gemini.py:33-34` raises `ProviderError` on empty content where `anthropic.py:28` returns `_Parsed("")` for the same condition — `base.py:139-141` documents both as valid, but the inconsistency is undeclared.
- `base.py:254` — `_configure` is an empty ABC method without `@abstractmethod` (ruff B027); a provider that forgets it fails later with `AttributeError` on `self._client`, surfaced as the misleading "API call failed".
- `base.py:217` / `cli_base.py:167` read `self._config.timeout_sec` directly, bypassing the public `timeout_sec` property added for this at `base.py:193`.
- Naive and aware datetimes mixed in one pipeline: `research/output.py:58,68` (naive local) vs `research/provider.py:16`, `research/cache.py:110` (UTC-aware).
- `orchestrator.py:225` — `saved_paths[0].stem[len("council-out-"):]` is unguarded prefix surgery; a stem that does not start with `council-out-` silently loses its first 12 characters instead of failing.
- `orchestrator.py:39` — `output_dir` parameter has no type annotation.
- `output.py:395` vs `:491` — two separate `_ts()` wall-clock reads per artifact; the header `**Date:**` and the filename timestamp can straddle a second boundary.
- Cost precision differs by surface: `research/display.py:67` (`.3f`) vs `research/output.py:94,154,162` (`.4f`) — a $0.0004 call reads "$0.000" live and "$0.0004" in the file.
- `tests/test_models.py:6-58` — the whole file constructs dataclasses and asserts the constructor stored the argument; passes against any dataclass definition.
- `tests/conftest.py:10-12` — the unit suite loads real global secrets from `~/Documents/.secrets/.env` at import.

---

# Architecture summary — four structural themes

### 1. The seams between well-tested units are where everything fails
The units are genuinely well built — `providers/base.py`'s template method, `seat_router.py`'s split of `try_cli`/`record_api` to avoid an import cycle, `crux_check.py`'s three-state parse. The failures are all at the joins: `_parse` outside the guard (**P1-1**), `gather` without `return_exceptions` (**P1-2**), clients built in `__init__` and used across loops (**P1-3**), two divergent precedence blocks in one function (**P1-4**), an unguarded mirror write inside a carefully-guarded router (**P1-5**). The test suite mirrors this exactly: every leaf is covered, and both orchestration functions are mocked out of existence (**P1-16**). **Recommendation: the highest-value new tests are integration tests at the two orchestration seams, not more unit tests.**

### 2. Two god-functions concentrate the complexity, and both produce their own bugs
`cli.run()` (complexity 50, 196 statements) and `CouncilRunner.run()` (complexity 20, 77 statements) hold the entire application flow between them. These are not merely long — the length is *causal*. P1-4's silent wrong-panel bug exists because two precedence blocks sit 120 lines apart in the same body; the dead `effective_synthesizer` at `cli.py:609` exists for the same reason. Everything else in `src/` is proportionate; extracting these two functions is the single highest-leverage structural change available.

### 3. Layering is enforced by convention and comment, not by structure
The intent is documented well and often — `runner.py:3-4` explains why `CouncilRunner` moved out; `seat_router.py:15-18` explains the split that prevents an import cycle; `crux_check.py:15-18` explains the ADR-03 signature guarantee. But nothing *enforces* it, and it has already leaked in three places: `research/output.py:10` imports the private `_write_routed` from a presentation module; `research/headless.py:25` imports `_error_result` from the Rich display module while its own docstring claims it touches no such machinery; `healthcheck.py:34` reaches into `provider._config` past a public property built for that purpose. `output.py` (**P1-13**) is where this concentrates — a 1,352-line module holding presentation, infrastructure, a markdown parser, and the inter-repo contract schema behind one name.

### 4. Error classification is string-matching all the way down, and it silently misroutes
Four independent classifiers — `classify_error` (`base.py:44`), `classify_cli_failure` (`base.py:100`), the research timeout sniff (`display.py:119`), the dissent/decision heading heuristics (`output.py:560-583,725-748`) — all operate on rendered strings. Three of the four demonstrably misfire: naked HTTP digits in request IDs (**P1-7**), a classifier reading a string this same module polluted with untrusted CLI stderr (`base.py:110` classifying the message built at `cli_base.py:210`), and a timeout check for a substring the providers never emit (**P1-11**). The structured data — SDK exception types, `status_code` — is available at every one of these sites and discarded. The markdown-heading heuristics are the defensible case (there is genuinely no structure to read), and notably they are the ones written most carefully.

**One thing worth saying plainly:** the comment quality in this codebase is unusually high, and the comments are load-bearing rather than decorative — `output.py:771-796` and `crux_check.py:40-61` record *why* a rule is shaped the way it is, naming the specific input that broke the previous version. That is real institutional memory. The failure mode it creates is the one this audit found repeatedly: **the comment states a guarantee the code does not implement** (**P2-20, P2-21, P2-22, P2-26**, and the phantom-path filter at **P2-4** that exempts its own artifact). A confident comment is now the *least* reliable signal in this repo — the invariants that hold are the ones with a test, and several of the most emphatic comments guard code with none.

---

# Buy-vs-build candidates

Ranked by hand-rolled line count against library maturity.

### 1. CommonMark inline parsing — ~270 lines · `output.py:771-1039`
`_pair_code_spans`, `_unwrap_emphasis`, `_top_level_bullets`, `_is_punctuation` plus five module-level regexes implement backtick-run pairing (CommonMark §6.1), left/right-flanking delimiter runs (§6.2), backslash escapes (§2.4), ordered/bullet list grammar (§5.2) and thematic breaks. `_unwrap_emphasis` alone is complexity 19 with 21 branches.

**The evidence this was the wrong build:** the comments document **six** successive correctness passes ("terra pass 2" through "pass 6") each fixing a payload-corruption bug — `- 3D printing` → `D printing`, `` `__init__` `` → `init`, an infinite loop on any Windows path, quadratic blowup on `" *a"` × 30k, NBSP-separated asterisks deleting a whole option. Every one of those is a solved problem in `markdown-it-py` (the CommonMark reference port, pure Python, actively maintained) or `mistune`.

**Recommendation:** parse the synthesis to an AST once with `markdown-it-py` and read headings, list items and inline text off the tree. This deletes the parser, `_split_sections`'s fence blindness (**P2-12**), and the `_one_line` emphasis-stripping idiom in one move. **Requires operator approval — new dependency.**

### 2. Retry / backoff — `debate.py:60-88`
The retry loop grows the *timeout* ×1.5 but never sleeps, so a `rate_limit` is re-fired immediately into the same 429. It also stacks on top of undisabled SDK retries (`max_retries` defaults to 2 in the openai/anthropic clients, never overridden in the debate lane), producing up to 6 HTTP attempts inside one `wait_for` budget sized for one call.

**Recommendation:** `tenacity` (exponential backoff + jitter + retry-on-exception-type, ~5 lines of decorator), plus `max_retries=0` on the debate clients so there is exactly one retry owner. Note the research lane already does the client half correctly (`perplexity.py:57`) — this is the debate lane lagging. **Requires operator approval — new dependency.**

### 3. Error classification — `base.py:44-124`
Two hand-rolled string classifiers, ~80 lines, demonstrably misfiring (**P1-7**). **No new dependency needed** — the openai and anthropic SDKs already ship typed exception hierarchies (`RateLimitError`, `AuthenticationError`, `APITimeoutError`, `APIStatusError.status_code`). This is buy-vs-build where the "buy" is already installed and being thrown away by calling `str(exc)`.

### 4. Responses-API extraction — ~150 lines triplicated across three providers
`openai_mini_research.py:110-161`, `openai_deep_research.py:113-162`, `grok_research.py:107-159`. Not a library candidate — an internal-module candidate. One shared `_responses_api.py` collapses three copies to one and makes a citation-extraction fix a single edit. Also removes the three self-inflicted `type: ignore`s.

### 5. Config schema validation — `config/config_loader.py`
Hand-rolled dataclass parsing with no schema validation, which is why `sdk`, `token_budget` and `perplexity.base_url` can be declared, typed, parsed, and never read with nothing detecting it (**P2-27, P2-28**). `pydantic` (or `pydantic-settings`) would make a declared-but-unconsumed key visible and a typo in `settings.yaml` a startup error rather than a silent default. **Requires operator approval — new dependency.** Lower priority than 1–3: the current loader works, the failure is silent dead config rather than incorrect behaviour.

### 6. Not worth buying — recorded so it is not re-litigated
- **HTTP** — the SDKs own the transport; there is no hand-rolled HTTP anywhere in the tree. Correct as-is.
- **The CLI layer** — Click and Rich are used idiomatically and well; `_DefaultGroup` (`cli.py:426-451`) is a 15-line shim that earns its place versus a dependency.
- **The dataclass model layer** — `models.py` is exemplary: pure data, no logic, no deps, and `CruxStatus(str, Enum)` at `:113` carries a comment explaining exactly why it inherits `str`. Do not "upgrade" this to pydantic.

---

## Appendix — what was NOT found

Recording the negatives so a future pass does not re-spend the tokens.

- **No circular imports** anywhere in `src/`. The graph is acyclic and the layering intent is real, even where it leaks (theme 3).
- **No bare `except:`** in the tree. The 10+ `except Exception` sites are deliberate and, with the exception of `research/display.py:123` (**P2-17**), correctly scoped with `# noqa: BLE001` and a rationale.
- **No hand-rolled HTTP, no hand-rolled JSON, no hand-rolled async primitives.**
- **No blocking I/O inside `async def` in the API or research lanes** — every network call is properly awaited. The two exceptions are both in `cli_base.py` (`:94-97` `subprocess.run` in the async kill path, `:148-152` the version probe) and are latent only because that subtree is unreachable from shipped config (**P2-32**).
- **No rotted test suppressions.** Exactly one `skip` exists in 11,421 lines of tests (`tests/test_integration.py:33`, module-level, keyed on API-key count). No `xfail`, no `skipif`, no commented-out tests. Nothing is hiding behind a marker.
- **`config/settings.yaml`'s `crux_check` block is a model of the pattern the rest of the config should follow** — `:406-409` explicitly documents why there is *no* `max_tokens` key ("would be dead config that reads as a bound but enforces nothing") and points at the code-side bound that actually applies. The dead keys in **P2-27/P2-28** are exactly what that comment was written to prevent.
- **The strongest test code in the repo is `tests/test_output.py:1342-1839`** — the `_top_level_bullets` corpus, with exact-equality assertions, two property-based fuzz guards (`:1693`, `:1722`), and complexity/termination guards for a scanner that provably hung. The three validator test files (`test_validate_sealed_keys.py:102-110`, `test_validate_docs_registry.py:242-251`) are equally honest, using real temp git repos and real subprocesses to exercise genuine fail-closed behaviour.

---

**Audit produced by:** Claude Opus 4.8, unattended night batch, read-only.
**No `src/` file was modified. No `BACKLOG.md` entry was filed. No item was struck.**
