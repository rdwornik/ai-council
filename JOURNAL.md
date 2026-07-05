# Journal — ai-council

### 2026-07-05 — ADR-11 + ADR-12 ratified; invocation contract authored; Wave-0 doc reconciliation (D4)

**Did:** Docs-only execution session on `docs/adr-11-12-ratification` (operator ratification authority granted in the session prompt; one checkpoint, approved with 4 adjustments). C1 `165b94a`: GUIDE Wave-0 reconciliation — all 4 example frontmatters aligned to the true default (dropped `models:` + `synthesizer: openai`; detection keys retained per inbox-sniffing convention), single labelled explicit-override example added, mechanism text corrected (synthesizer is EVICTED from the panel, not swapped), explicit "effective default = 4 debaters + gemini non-participating synthesizer" statement added. C2 `68d0571`: ADR-11 (delegated invocation contract) ratified — Accepted (2026-07-05), Related carries the fleet-recon reconciliation line (D1–D3 HOLD). C3 `9938161`: `protocols/COUNCIL_INVOCATION_CONTRACT.md` authored — lanes A/B, flag set verified against live cli.py, frontmatter precedence, exit codes 0/1/2/3 with caller obligations, artifacts, JSON payload, degradation/RoutingError semantics, hub-WHEN/WHY vs protocols-HOW boundary, Lane-A caller walkthrough, MANDATORY Known-deviations section naming both D2 parity gaps (`--file` frontmatter leak; research `--return-dir` no-op). C4 `c11fb42`: ADR-12 ratified with the fleet-recon §7 markup applied verbatim (v1 adapters = claude+codex; four witnessed safety invariants incl. scratch-cwd-primary-isolation and identity-or-no-seat; gradient codex > claude > grok(post-OAuth) > agy(excluded); per-call pin rule; §5 default-flip stays evidence-gated; gemini seat path struck). C5 `3325a8a`: README index +2 Accepted rows; VISION References line unbound (ADR-01 onward → index pointer, cannot re-stale at ADR-13).
**Result:** Zero source-code changes (`git diff --stat main` = protocols/ + docs/decisions/ + VISION.md + JOURNAL.md only); pytest unit suite + ruff green — 426 passed (first full run flaked once on `test_inbox_exits_3_when_any_batch_run_degraded`, which passes in isolation and on the full re-run; ordering-dependent, unrelated to a docs diff — watch for recurrence); all pre-commit gates passed per commit (TOC freshness on the GUIDE edit, canonical_freshness on VISION). ADR-13 untouched (still a draft inside the 07-04 audit only). BACKLOG untouched — precedent verified (ADR-09's BACKLOG edit closed #12; ratifying ADR-12 closes nothing, #16 remains open until a CLI backend runs a debate turn). Status divergence recorded for a future hygiene pass: ADR-09/10 still say Proposed while 11/12 say Accepted. Deferred staleness named: CLAUDE.md §11 still lists local ADRs only through ADR-08 (its own currency pass; edits there trip the canonical-freshness gate).
**Changes:** `protocols/COUNCIL_QUESTION_GUIDE.md`, `docs/decisions/ADR-11-delegated-invocation-contract.md` (new), `protocols/COUNCIL_INVOCATION_CONTRACT.md` (new), `docs/decisions/ADR-12-provider-backend-engine-and-cost-lanes.md` (new), `docs/decisions/README.md`, `VISION.md`, `JOURNAL.md` (this). Commits `165b94a`/`68d0571`/`9938161`/`c11fb42`/`3325a8a` + this entry, merged `--no-ff` to main. NOT pushed. Next: D2 parity fixes (separate pause-gated code session) close the contract's Known-deviations; hub-side session for the D14 flags.

---

### 2026-07-05 — Fleet recon, liveness & process design persisted to docs/audits (operator close-out `2593075`)

**Did:** Ran the consolidated fleet-recon session (operator-approved probe matrix + 4 amendments): live-probed all 5 agentic CLIs from a scratch cwd (claude 2.1.200, codex 0.141.0, agy 1.0.16, grok 0.2.82, deepcode 0.1.33, + legacy gemini 0.49.0 as Step-4 evidence), ran Step-2 liveness via the council's own `run_health_checks` on verbatim `settings.yaml` pins, swept model currency against live provider lists, reconciled the 2026-07-04 Fable audit and the architect's browser research against witnessed state, and delivered 4 functional process specs (doctor / lane routing / delegation lifecycle / debate lifecycle) + ADR-12 markup + 12-fork list. Report committed as `docs/audits/2026-07-05-fleet-recon-liveness-and-process-design.md` (SHA `2593075`).
**Result:** Witnessed 5-CLI recon: v1 CLI adapters = claude+codex; agy excluded by identity roulette (silent model-pin swap, no identity channel); deepcode non-headless (TTY-required); grok seat-capable but API-billed (no OAuth configured); legacy gemini auth-dead (consumer shutdown). Liveness 9/9 PASS (Anthropic credits healthy) + one stale research pin (`grok-4.20-reasoning` → `grok-4.20-0309-reasoning`; NOT changed — operator decides). Fable-audit D1–D14 + corrections #1–#5 all HOLD (one embedded ADR-12 premise invalidated: grok/deepseek CLIs DO exist). Safety facts witnessed: `claude -p --tools ""` still ingests cwd CLAUDE.md; `codex exec` hangs on open stdin. Zero secrets in captures (scanned).
**Changes:** `docs/audits/2026-07-05-fleet-recon-liveness-and-process-design.md` (new, `2593075`), `JOURNAL.md` (this). Direct-to-main close-out commits per explicit operator instruction (supersedes the branch→merge rule for this wrap only); no push; no config edits.

---

### 2026-07-04 — Fable architecture audit persisted to docs/audits (operator instruction)

**Did:** Persisted the 2026-07-04 Fable architecture audit as `docs/audits/2026-07-04-fable-architecture-audit.md` — current-state vs consolidation-brief gap analysis across 5 areas (invocation surface, backend/cost model, epistemic mechanics, process ownership, end-to-end pipeline), draft ADR-11 (delegated invocation contract) / ADR-12 (provider backend engine + cost lanes) / ADR-13 (bounded crux-check, baseline-gated), and tagged decision list D1–D14. The audit itself ran earlier the same day in the `fable-audit` worktree under MODE PLAN (zero repo changes; deliverable emitted to the session plan file); the operator ordered persistence after review. Added a Status/provenance header on the copy; ADR texts remain drafts — `docs/decisions/` unchanged.
**Result:** Doc-only change; no code or tests touched. Load-bearing verified findings recorded in the audit: the "de-facto openai synthesizer" premise is false in code (`exclude_synthesizer_from_panel()` runs before `pick_synthesizer()`, so gemini IS the runtime default) but real in practice via the guide's `synthesizer: openai` frontmatter examples; the default full panel de-facto debates with 4 models; `--file` mode skips frontmatter parsing; research mode ignores `--return-dir`; the Epic C CLI-flag recon does not exist in-repo. Housekeeping same session: `fable-audit` worktree removed safely (branch had zero unique commits, tree clean; branch deleted via `-d`); leftover empty dir `.claude/worktrees/fable-audit` removable post-session (locked by the session process cwd on Windows).
**Changes:** `docs/audits/2026-07-04-fable-architecture-audit.md` (new), `JOURNAL.md` (this). Branch `docs/fable-audit`, merged `--no-ff`.
**Next:** Operator + primary-chat architect review the ADR-11/12/13 drafts; if ratified, Wave 1 (ADR-11: `protocols/COUNCIL_INVOCATION_CONTRACT.md` + `--file` frontmatter parity + research `return_dir` parity) is the first implementation session — baseline-independent. Epic B #1 should adopt the D5 scoring guard (verify per-transcript verdict author before scoring).

---

### 2026-07-02 — Epic A output subsystem: return_dir routing (#13) + double-council fix (#14) + minority report (#15)

**Did:** Implemented the three Epic A output-subsystem items on `feat/output-subsystem`, per-item commits. **#13 (`bfc268f`):** ADR-10 deterministic-return routing — `--return-dir <path>` CLI flag + `RunRequest.return_dir`, threaded through both interactive and inbox paths; new `output._write_routed()` centralizes canonical + secondary + return + target writes (canonical `./output/` always fires first, return_dir auto-mkdir + best-effort); reserved seam left for the `~/.claude` `council.return_dir` reader (deferred per ADR-10). **#14 (`53ad525`):** `clean_slug()` now strips one leading "council" token so inbox files no longer emit `council-out-…-council-…` (bare `council`/`councillor` preserved). **#15 (`f1a4b74`):** `extract_dissent()` + `save_minority_report()` emit a discrete `council-minority-<ts>-<mode>-<slug>.md` artifact on a non-unanimous verdict, routed to the same destinations as the verdict (Rama 4).
**Result:** `.\scripts\check.ps1` — 426 unit tests pass, ruff clean; mypy shows exactly the 6 pre-existing #20 provider errors (zero new; touched files mypy-clean) → diff-scoped gate satisfied. Empirical done-contract driven through the real CouncilRunner→synthesis→output path with mocked providers (no API spend): #13a no-flags → `./output/` only, not `.dev-knowledge/`; #13b `--return-dir` → routed copy + canonical both written, same filename; #14 `council-question-x.md` → single-"council" filename; #15 dissent → separate durable minority artifact alongside the verdict.
**#15 trigger (architect-approved):** "non-unanimous final vote" operationalized as a substantive dissent section in the synthesizer's verdict (Unresolved Disagreements / Contested Points / explicit dissent), since ai-council has no structured vote tally (ADR-03 voting is free-text) — no Council runtime behavior changed.
**Changes:** `src/ai_council/{output.py,inbox.py,cli.py,orchestrator.py,models.py}`, `tests/{test_output.py,test_inbox.py}`; commits `bfc268f`, `53ad525`, `f1a4b74` on `feat/output-subsystem`. Refs ADR-10, BACKLOG #13/#14/#15.
**Next:** Merge `feat/output-subsystem` → `main` (`--no-ff`). Deferred (unchanged): the `~/.claude` `council.return_dir` reader (ADR-10 reserved seam), BACKLOG #9 ADR-67 question-quality pieces.

---

### 2026-06-03 — ADR-71 rollout: consume hub TOC hook + add TOC to council-question-guide

**Did:** Wired ai-council as the second consumer of the `.dev-knowledge` hub TOC hook (ADR-71 pinned-pull, rev `69558c7`) — added `repo: ../.dev-knowledge` stanza scoped to `docs/council-question-guide.md` with `toc-freshness` (gate, pre-commit) + `toc-generate` (manual); ran `toc-generate` to produce a 29-entry TOC in the guide; confirmed gate passes-fresh / fails-stale.
**Result:** 407 unit tests unchanged; ruff clean; `toc-freshness` Passed on current file, exit 1 on stale edit. Codemap not touched (gated on BACKLOG #79 — ai-council's frozen codemap hand-authored status ungrounded).
**Changes:** `.pre-commit-config.yaml` (+hub TOC stanza, +`default_stages: [pre-commit]`), `docs/council-question-guide.md` (+32 lines TOC), `JOURNAL.md` (this). 2 commits on `feat/consume-toc-hook`; merged `--no-ff`.

---

### 2026-06-02 — Universalization coherence audit (G1): doc-truth conformance to current standard

- **Did:** Ran the per-child-repo universalization coherence audit against `.dev-knowledge` committed `main` (read-only). Confirmed machine floor via imported `scripts/audit.py` `audit_repo` (not the CLI). Fixed doc-truth drift the 10 checks don't cover: ARCHITECTURE.md (post-ADR-38 namespace path `src/research/`→`src/ai_council/research/`; Folder Governance aligned to ADR-60 child-repo taxonomy — dropped stale `handoffs/` + never-existed `eval/` rows; `last_reviewed` re-stamped after end-to-end re-read) and CLAUDE.md (added `last_reviewed` frontmatter; PLAYBOOK path; §7/§8 reconciled to actual `~/.claude/`+`.claude/` state; §10 namespace path; §11 +local ADR-08, +ecosystem ADR-59/60/67).
- **Result:** Machine floor 9/10→**10/10 pass, 0 fail, 0 warn** (cleared the canonical_freshness FAIL the 2026-05-28 mermaid commits had introduced + the CLAUDE.md no-frontmatter WARN). 407 unit tests pass unchanged. Doc-only; no Codex gate.
- **Decisions (operator):** **D1 = Defer** — keep ai-council's ADR-41/47 BACKLOG stream schema; do NOT migrate to the ADR-64/65/66 story-map. Cascade of ADR-64/65/66 to child repos is unresolved upstream (`.dev-knowledge` BACKLOG #20 open); deferred to the canonical-baseline decision. CLAUDE.md §11 pending note left as-is. **D2 = Track, don't build** — added one open BACKLOG item (Governance stream) capturing the ADR-67 implementation obligation (`/council-question` template + question-gate + `council.return_dir`); downstream, not built now.
- **Changes:** `ARCHITECTURE.md` (15add98), `CLAUDE.md` v2.2 (690b326), `BACKLOG.md` (+1 ADR-67 item), `JOURNAL.md` (this entry).
- **Abandoned:** BACKLOG schema migration (D1 = Defer — would pre-empt an open upstream decision).
- **Next:** Merge `chore/universalization-conformance` → `main` (`--no-ff`); delete branch. ADR-67 implementation stays deferred.

---

### 2026-05-19 — ADR-53 chunk 4: AGENTS.md retired, CLAUDE.md v2.1 live

- **Did:** Executed ADR-53 chunk 4 — full migration of `ai-council/AGENTS.md` content into a single canonical `CLAUDE.md` v2.1 (139 lines, ≤200 cap). Displaced technical depth (architecture tree, key commands, design decisions, transcript routing, debate modes, research providers, folder governance, inbox detection) moved to `ARCHITECTURE.md` (6 new `[L-opt]` sections, ADR-51 conformant) and `README.md` (3 missing CLI examples). Stale test count ("266 unit tests") removed from `.claude/rules/testing.md` (local-only; `.claude/` is gitignored). Moot BACKLOG.md P3 item (AGENTS.md creation) removed. AGENTS.md deleted.
- **Result:** `ai-council` now has a single 12-section CLAUDE.md agent-instruction file per ADR-53. ARCHITECTURE.md fully populated per ADR-51. No Python touched; 407 unit tests unchanged. AGENTS.md historical references in JOURNAL, LESSONS, and audits left intact as immutable records.
- **Changes:** `ARCHITECTURE.md` (+6 sections), `BACKLOG.md` (moot item removed), `README.md` (3 CLI examples), `CLAUDE.md` (full v2.1 rewrite, 139 lines), `AGENTS.md` (deleted).
- **Abandoned:** nothing.
- **Next:** Merge `docs/chunk4-ai-council-claude-md-migration` → `main`; run Phase 2 smoke test (Step 5 in BACKLOG, Synthesizer Refresh stream).

---

### 2026-05-19 — ADR-51/52 conformance (ARCHITECTURE.md + AGENTS.md §7)

- **Did:** Created `ARCHITECTURE.md` from the ADR-51 canonical template (Purpose, Codemap with `<!-- CODEMAP:START/END -->` markers, Layer Boundaries & Invariants, Data Flow); added ADR-51 + ADR-52 to `AGENTS.md` §7 ecosystem ADR list; bumped `Last updated` stamp to 2026-05-19.
- **Result:** `ai-council` fully conformant with ADR-51 and ADR-52. 407 unit tests pass unchanged.
- **Changes:** `ARCHITECTURE.md` (new), `AGENTS.md` (§7 + Last updated stamp).
- **Abandoned:** nothing.
- **Next:** corp-ops trigger-based rollout (separate task per audit).

---

### 2026-05-18 — Perplexity research-provider timeout fix

Research run reported `perplexity ✗ timeout 1m 00s`. One live reproduction with a 300s ad-hoc ceiling (real council research brief through the actual provider code path) completed cleanly in **68.2s** with 25.7k chars and 8 sources — confirming Perplexity itself is healthy and the 60s ceiling was simply too tight. Audit (2026-05-18) had already flagged Perplexity as the only research provider still on the old single-shot pattern (no SDK retry, no SDK timeout).

**Result:** Raised `research.providers.perplexity.timeout_sec` from 60 → 240 in `settings.yaml`, and passed `timeout=` + `max_retries=1` into `AsyncOpenAI` so the SDK enforces request lifetime and owns a single transient retry — Fix-A parity with `openai_mini_research.py`. Post-fix live verification: 69.1s clean. Outer `asyncio.wait_for` retained as the hard cancellation guard. Added a regression test asserting both the SDK-level kwargs and the configured 240s value.

**Changes:** `config/settings.yaml`, `src/ai_council/research/providers/perplexity.py`, `tests/test_research.py`.

---

### 2026-05-18 — Claude billing-condition diagnosis + mode-scoped health gate

Operator reported `council --inbox -M r` blocked at startup by health-check failures on `claude` / `claude-sonnet` (HTTP 400 from `api.anthropic.com`). Single live reproduction with full body capture isolated the cause as account-level: Anthropic returns `400 invalid_request_error` with message `"Your credit balance is too low to access the Anthropic API"` when the org is out of credits — not a code bug, not a stale model alias, not an SDK / `anthropic-version` mismatch. Model strings `claude-opus-4-7` / `claude-sonnet-4-6` and the request envelope were all accepted by the server. Git evidence: neither `fix/openai-research-provider-migration` nor `fix/research-panel-degradation-alarm` touched `claude` config, `anthropic.py`, or `healthcheck.py` — operator hypothesis falsified.

**Result:** Two follow-up code fixes (since the billing condition is operator-handled out of band): (1) `classify_error` now recognises billing exhaustion (Anthropic `credit balance is too low` + OpenAI `insufficient_quota`) as a distinct non-retryable `"billing"` category with a clear health-check message — the prior `"invalid request during health check"` was opaque and misled diagnosis. (2) `council -M r` (research) now health-checks only the summarizer (`deepseek` by default), non-blocking; the merger's existing truncation fallback (`research/merger.py:184-186`) means a summarizer outage warns but never blocks retrieval. Debate modes preserve the full-pool blocking gate. Decision lives in a small testable helper `_select_health_check_targets`.

**Out of scope / known:** Two `tests/test_research.py::TestDegradationCLIExitCode` cases fail in the full suite for the same billing condition (they make live `claude` API calls and take 5 min each) — pre-existing, resolves on top-up; not marked `@pytest.mark.integration` today.

**Changes:** `src/ai_council/cli.py`, `src/ai_council/providers/base.py`, `src/ai_council/healthcheck.py`, `tests/test_cli.py`, `tests/test_base_provider.py`, `tests/test_healthcheck.py`.

---

### 2026-05-18 — Research-panel degradation alarm + provider doc reconciliation

Closed the systemic finding from the 2026-05-18 health-check audit by adding a loud aggregate alarm: when fewer than `min_successful_providers` succeed (default 3, denominator = selected panel including build-time dropouts), the research run still completes but emits a banner in console + saved markdown and the CLI exits with code 3 (distinct from Click's reserved 2). Decision recorded as ADR-08. Verified the configured Gemini agent ID `deep-research-preview-04-2026` is accepted at runtime via one minimal live `interactions.create()` call; CLAUDE.md Gotcha entry updated. Reconciled CLAUDE.md Grok provider-table row to match `settings.yaml` (`grok-4.20-reasoning`).

**Changes:** `config/config_loader.py`, `config/settings.yaml`, `src/ai_council/cli.py`, `src/ai_council/research/{merger,models,output,runner}.py`, `tests/test_research.py`, `CLAUDE.md`, `docs/decisions/ADR-08_research-degradation-alarm.md`.

---

### 2026-05-18 — OpenAI research-provider migration

Migrated `openai_mini` and `openai_deep` off the deprecated `o4-mini-deep-research` / `o3-deep-research` models onto the current `gpt-5.4-mini` / `gpt-5.5` + `web_search` Responses-API path. Sync call (background+poll dropped), single-shot retry on transient APIError, annotation-based parsers, real per-1M pricing in settings. Pre-migration live call confirmed `o4-mini-deep-research` returns `status=failed`; post-migration live calls verified non-empty content AND sources for both providers.

**Changes:** `src/ai_council/research/providers/openai_mini_research.py`, `src/ai_council/research/providers/openai_deep_research.py`, `config/settings.yaml`, `tests/test_research.py`, `scripts/verify_openai_mini.py`, `scripts/verify_openai_deep.py`.

---

### 2026-05-18 — Research-provider health check

Diagnosed the five research providers; flagged `openai_mini` (likely deprecated `web_search_preview` tool name) and `grok` (model string `grok-4.20-reasoning` mismatches `CLAUDE.md` and may not resolve) as at-risk; also surfaced `openai_deep` (no search tool passed) and a `gemini` agent-ID mismatch. Report is evidence, not a fix.

**Changes:** `docs/audits/2026-05-18-research-provider-health-check.md`.

---

### 2026-05-17 — Research-mode format in question guide

`council-question-guide.md` now gives `research` mode its own retrieval-brief format (`### Background` / `### What to find out` / `### Source rules` / `### Output wanted`); decision-mode sections scoped with blockquotes pointing to the new format.

**Changes:** `docs/council-question-guide.md` (research-mode format + decision-mode scoping notes).

---

### 2026-05-17 — Context-section danger-zone callout in bias guide

Added a Context-section danger-zone callout to the question-framing bias guide, driven by evidence from the 2026-05-17 bias audit that framing failures cluster almost entirely in the Context section rather than the headline.

**Changes:** `docs/council-question-guide.md` (new Context-section subsection).

---

### 2026-05-17 — F-0 fix: preserve full question in pick/judge transcripts

Pick/judge debate transcripts now embed the full submitted question text in a `## Question` section, at parity with research-mode output. Previously only a 70-80 char truncated H1 title was preserved, with the `Source:` field pointing at an external file that might no longer exist — making question framing unrecoverable. Forward-only; no backfill of past transcripts.

**Changes:** `src/ai_council/output.py`, `tests/test_output.py`.

---

### 2026-05-17 — Question-framing bias audit

Audited 21 past curated Council debate questions against the question-framing bias rubric in `docs/council-question-guide.md`; 9 research-mode questions scored in full, 12 pick/judge headlines scored at title-only (full prompt not preserved in transcript). Asker-leakage, loaded terminology, and anchoring dominate; report is evidence for an operator decision, not a recommendation.

**Changes:** `docs/audits/2026-05-17-question-framing-bias-audit.md` (new audit report).

---

### 2026-05-17 — Question-framing bias-elimination section

**Did:** Added a cross-mode question-framing bias-elimination section to `docs/council-question-guide.md`, covering seven framing biases, a pre-flight self-check, and a research-mode sharpener.

**Changes:** `docs/council-question-guide.md` (new bias-elimination section).

---

### 2026-05-17 — Research-mode question guide + AGENTS.md

**Did:** Added a "Research-mode questions" section (recognition test + formulation rules + breadth-over-depth trap) to `docs/council-question-guide.md`; created `AGENTS.md` at repo root from the canonical ecosystem template (`.dev-knowledge/templates/AGENTS-md-template.md`) per Council #28.

**Result:** 362 tests green. Branch `docs/research-mode-guide-and-agents-md` ready for review.

**Changes:** `docs/council-question-guide.md` (new research-mode section); `AGENTS.md` (new file).

---

### 2026-05-17 — Documentation simplification rollout (ADR-48/49/50)

**Did:**
- Created branch `feat/docs-simplification-rollout`
- Removed `CHANGELOG.md` and `BACKLOG_ARCHIVE.md` per ADR-49
- Copied `scripts/normalize_headers.py` from `.dev-knowledge`; ran it over LESSONS.md (no-op — already H3 pipe schema) and JOURNAL.md (H2 → H3 dated entries)
- Added `.pre-commit-config.yaml` wiring normalize_headers as a local pre-commit hook
- Added "Documentation conventions" section to `CLAUDE.md` (no CHANGELOG, no BACKLOG_ARCHIVE, Conventional Commits standard, JOURNAL/LESSONS structure)
- Added transcript-to-ADR workflow step to `docs/council-question-guide.md`

**Result:** 362 tests green. Branch `feat/docs-simplification-rollout` ready for review. Not merged, not pushed.

**Changes:** CHANGELOG.md deleted; BACKLOG_ARCHIVE.md deleted; JOURNAL.md header levels H2→H3; CLAUDE.md +11 lines; council-question-guide.md +7 lines; scripts/normalize_headers.py added; .pre-commit-config.yaml added.

**Abandoned:** Step 4 (LESSONS ordering) — already reverse-chronological, no action needed.

**Next:** Operator reviews branch and merges if satisfied. Then apply same rollout to `corp-ops` and `corp-sca-time-automation`.

---

### 2026-05-15 — ADR-46+47 compliance cleanup (cross-repo handoff)

**Did:**
- LESSONS.md: migrated `## Session: Phase 1 Foundation (2026-02-21)` → `## 2026-02-21` + Session label in body
- JOURNAL.md: moved 2026-05-12 addendum entry to correct reverse-chrono position
- BACKLOG.md: [blocked] → [open] + Blocked annotation on Step 6; Status field added to all 11 entries; BACKLOG_ARCHIVE.md created
- Driven by .dev-knowledge cross-repo audit (2026-05-15-ecosystem-audit.md) + handoff bundle
- LESSONS.md H3 entries re-ordered to reverse-chrono (follow-on: 2026-05-12/2026-05-11 entries appeared after April entries)

**Result:** ai-council compliant with ADR-46 + ADR-47. Re-audit from .dev-knowledge expected to clear all 5 FAIL checks.

**Next:** Operator runs `python scripts/audit.py run` in .dev-knowledge to confirm. Stream B P1 items flip to [done] on clean audit.

---

### 2026-05-13 — P3 BACKLOG entry captured for ADR-34 timestamp-underscore case

**Did:** Added P3 BACKLOG entry naming the specific case (council-out filename `YYYYMMDD_HHMMSS` timestamp underscore) and the methodology question (ISO timestamp exempt from ADR-34?); cross-linked to existing P2 CI enforcement entry.

**Failed:** —

**Next:** Methodology decision on ADR-34 ISO-timestamp exemption — can be addressed when ADR-45 implementation surfaces it OR sooner if convenient.

---

### 2026-05-12 — Scrum-master review implementation (.dev-knowledge strażnik)

**Did:**
- Implemented 9 of 10 findings from `.dev-knowledge` scrum-master review (2026-05-12)
- C1: retired `tasks/todo.md` (255 vs 362 test stale + March 2026 checklist); surviving items migrated to BACKLOG.md
- I1: created `BACKLOG.md` per ADR-41 schema (8 streams, 11 items seeded)
- I2+I3: `README.md` architecture section updated to `src/ai_council/` namespace layout + test count to 362
- I4+I5: `docs/COUNCIL_QUESTION_GUIDE.md` → `docs/council-question-guide.md` and `docs/SYNTHESIS-QUALITY-RUBRIC.md` → `docs/synthesis-quality-rubric.md` (ADR-34 hyphen+lowercase)
- I6: `2026-03-15_CODE_REVIEW_REPORT.md` + `2026-03-26_CODE_REVIEW_REPORT.md` archived to `docs/audits/archive/legacy/`
- M1: `VISION.md` `last_reviewed` bumped 2026-05-09 → 2026-05-12
- M3: 4 lessons appended to `tasks/lessons.md` (target resolver fail-loud, inbox parity 3rd instance, ADR-43 schema DRYness, observability field design)
- M2 (AGENTS.md addition) deferred per strażnik own "low urgency" framing; tracked in BACKLOG.md P3 Governance

**Result:** ai-council fully aligned with strażnik audit findings except deferred M2. Audit pattern validated — I5 fresh violation caught and fixed same-day. CHANGELOG + commits = audit trail per single-round-trip principle.

**Next:** Step 5 smoke test (operator-driven, BACKLOG P1 Phase 2).

---

### 2026-05-12 — Phase 1 + ADR-34 hyphen combined

**Did (Phase 1):**
- Per-synthesis observability emitted: latency, transcript size, timeout flag, output tokens, error class — `DebateResult.synthesis_metrics` + `_metrics.json` synthesis block
- Created `docs/SYNTHESIS-QUALITY-RUBRIC.md` (5-point operator checklist)
- ADR-06 Qwen trial closed-out: deferred/abandoned with reopen trigger (DeepSeek round-blocking >2%)
- Gemini synthesizer version check: Case A — already on `gemini-3.1-pro-preview` (3.x), no upgrade action

**Did (ADR-34):**
- Council CLI emitter format flipped to hyphen per `.dev-knowledge` cycle 2 ratified mandate: `council_out_*` → `council-out-*`
- Downstream patterns updated (tests + docs aligned); no historical transcript rename (pre-decision artifacts)

**Result:** Observability foundation in place for smoke test (Phase 2). Cross-repo cycle 2 Change 1 implementation complete.

**Next:** Turn 4 delivery report to `.dev-knowledge` for cycle 2 closure; then Phase 2 smoke test operator-driven execution once baseline reads accumulated.

---

### 2026-05-12 — Scrum-master addendum implementation (I7 + I8)

**Did:**
- I7: moved `tasks/lessons.md` → `LESSONS.md` at repo root; retired `tasks/` folder entirely
- I8: renamed `docs/handoffs/_archive/` → `docs/handoffs/archive/`
- CLAUDE.md updated (Lessons Discovery bullet + Folder Governance `tasks/` entry replaced with `LESSONS.md`)
- VISION.md lessons path reference updated
- BACKLOG.md: no separate LESSONS.md-absent item existed; AGENTS.md M2 remains open (deferred)
- LESSONS.md: architect-side lesson captured on local-config-defense failure mode

**Process:** Both findings caught by operator post main-review implementation. Single-branch, 4 commits. Historical entries in CHANGELOG/JOURNAL left immutable.

**Result:** ai-council fully aligned with ecosystem convention on lessons location + archive folder naming. Original 10 findings + 2 addendum findings = all addressed except AGENTS.md (M2 from main review, still deferred per strażnik "low urgency").

---

### 2026-05-11 — ADR governance sweep + HANDOFF cleanup

**Did:**
- Audit ADR-01..07 status headers against current ecosystem state
- ADR-07: file status flipped to "Superseded by ADR-43" — was index-only before today; file is source of truth
- ADR-01: status date updated to 2026-04-30 (Gemini synthesizer revision); header had captured only the 03-29 Sonnet revision
- ADR-02: revised to reflect 5-model default panel; original "3-model default" was factually wrong per current CLAUDE.md and code
- ADR-05: provider count corrected 3→4 (Grok/XAI added post-ADR, undocumented in ADR body)
- ADR-06: Qwen trial marked deferred (not pending); Gemini synthesizer change cross-referenced to ADR-01
- ADR-03, ADR-04: verified current, no changes
- decisions/README.md: index re-synced with ADR-01, ADR-02, ADR-06 updated statuses
- HANDOFF.md: deleted — handoff process owned by `.dev-knowledge` per ADR-42; pointer file adds noise not value

**Result:** ADR status headers are now authoritative in files; index mirrors them. Governance docs internally consistent.

**Candidates for future work (from audit):**
- ADR-01 Synthesizer selection: Gemini default still operative; model landscape has evolved (Claude 4.7, Gemini 3.x era). Candidate for meta-debate: should default panel + synthesizer refresh for 2026 model landscape?
- ADR-06 Cost optimization: Qwen trial deferred indefinitely; OpenRouter hedge not implemented. If DeepSeek reliability degrades again, Qwen/OpenRouter question will resurface.

---

### 2026-05-11 — Docs hygiene sweep

**Did:**
- Five-file docs internal-alignment pass post today's feature work
- HANDOFF.md: replaced pre-ADR-42 feature status doc with pointer to .dev-knowledge-owned handoff process
- COUNCIL_QUESTION_GUIDE.md: added `target-project` frontmatter + `--target-project` CLI flag section
- decisions/README.md: complete index (ADR-01 through ADR-07 with status) + cross-repo ADR-43 reference
- docs/archive/ consolidated into docs/audits/ with git history preserved via `git mv`
- docs/audits/README.md: new convention doc

**Result:** Internal docs reflect current state across all feature work shipped today. No code, test, or config changes.

---

### 2026-05-11 — ADR-43 amendment cycle 1 implementation

**Did:**
- Refactored `target_projects` schema per `.dev-knowledge`-approved ADR-43 amendment: `dev_root` + opt-in name list, paths computed as `<dev_root>/<name>/docs/decisions/transcripts/`
- Updated `TargetResolver` constructor signature and path computation; updated cli.py caller
- Adjusted ~10 existing test cases; added 5 new validation tests (dev_root required, dir validation, dict migration error, duplicate names, path computation) — 359 total
- Updated README.md + CLAUDE.md with new schema examples and ADR-43 reference
- Archived `.dev-knowledge` cycle closure note for symmetric audit trail
- Codex `/review` pending

**Result:** Schema is DRY; ecosystem root declared once; new repos join routing via single-line list addition.

**Next:** Codex `/review`; then generate delivery report for `.dev-knowledge` (Turn 4 implicit closure of cycle 1 handshake). Operator decides `git push` timing.

---

### 2026-05-11 — Post-routing cleanup

**Did:**
- Disabled `secondary_output_enabled` default — resolves architectural overlap with new `target_paths` per-invocation routing
- Added README Transcript Routing section (closes acceptance-criteria miss from previous session)
- Fixed CLAUDE.md test count drift (349 → 354)

**Result:** Clean post-routing state. No double-write to `.dev-knowledge` when `--target-project .dev-knowledge` used; README documents the feature for users.

**Next:** `.dev-knowledge` ESSENTIALS update (separate session). `git push` when ready (currently 21+ commits ahead of origin).

---

### 2026-05-11 — Cross-project transcript routing (feat/transcript-routing)

**Did:**
- Implemented opt-in, config-driven per-invocation transcript routing for all 4 modes
- Added `target_projects` map to `config/settings.yaml` + `AppConfig` loader with validation
- Created `src/ai_council/routing.py`: `TargetResolver` + `RoutingError` (fail-loud on unknown names)
- Extended `inbox.py` `parse_file` to accept optional resolver, resolve `target-project` frontmatter at parse time
- Added `target_paths: list[Path]` parameter to `save_to_file` and `save_research_to_file` — auto-mkdir, best-effort mirror
- Added `--target-project` Click flag (multiple=True) to CLI, wired through RunRequest → orchestrator → output
- 6 commits on branch `feat/transcript-routing`; 349 tests pass; ruff at pre-existing 17 errors baseline

**Architecture decisions:**
- Names dynamic (frontmatter / flag), paths static (settings.yaml) — two-layer model per spec
- Single `TargetResolver` called from both CLI flag path and inbox frontmatter path — no forked logic
- Canonical write always first (hard); mirror writes best-effort with logging
- Existing `secondary_dir` behavior unchanged — coexists with new `target_paths`

**Next:**
- `.dev-knowledge/protocols/ESSENTIALS.md` "Council output convention" section update — separate `.dev-knowledge` session
- Await operator confirmation to merge `feat/transcript-routing` → main

### 2026-05-09 — Audit-sync governance closure (F-01, F-02)

**Did:**
- Verified prior commit `62c1f7d` (config/settings.yaml grok model `grok-4.20 → grok-4.3`) matches Stage 3 expected pattern; commit was made by a prior session, not this one
- Created `VISION.md` (tier M, ADR-33 Lite: Mission / Scope / Relationships / Lifecycle)
- Configured lessons discovery in `CLAUDE.md` (`DEV_KNOWLEDGE_PATH` env var per ADR-35)
- Updated CHANGELOG

**Result:** F-01 + F-02 closed. Baseline 310/310 tests passing. Branch `docs/audit-sync-2026-05-09` ready for review and merge (3 commits ahead of main).

**Next:** return `09_EXECUTION_EVIDENCE.md` to .dev-knowledge for review. Await ADR-40 recalibration before tackling F-03 (BACKLOG.md) and F-04 (ARCHITECTURE.md).

### 2026-04-30 | ADR-38 migration: src/ → src/ai_council/
- Moved all 34 source files under `src/ai_council/` via `git mv` (history preserved); rewrote 73 internal imports in src/, 83 imports + 56 mock.patch string literals in tests/
- Updated pyproject.toml: added `[build-system]` (`setuptools.build_meta`), `where=["src","."]` for packages.find, new entry points, coverage paths; deleted pytest.ini (consolidated into `[tool.pytest.ini_options]`)
- 310 unit tests pass, identical to pre-migration baseline; zero functional changes

### 2026-04-24 | Fix research providers (Gemini 404, OpenAI mini 400)
- Gemini research: `gemini-2.5-pro-preview-05-06` → `gemini-2.5-pro` (preview was not yet released)
- OpenAI mini: added `tools=[{"type": "web_search_preview"}]` to Responses API call (deep research models require at least one search tool)
- Full smoke test: Perplexity + Gemini both completed; OpenAI mini job accepted + completes (~3min for simple queries, may be transient-fail on complex topics)
- 255 tests passing

### 2026-03-29 | Sonnet 4.6 synthesizer + mypy CI
- Added `claude-sonnet` provider; set as default synthesizer (5x cheaper than Opus)
- mypy CI enforcement via `scripts/check.ps1` (pytest + mypy + ruff, 0 errors)
- Archived code review reports to `docs/archive/`
- 255 tests

### 2026-03-28 | Retry logic + graceful degradation
- Error classification (`classify_error()`), `was_retry` tracking
- Specific healthcheck messages per provider failure mode
- `RunPolicy` (retry_on patterns, min_panel_size) decoupled from debate logic
- 231 → 255 tests after provider unit tests + orchestrator extraction
- Next: Sonnet synthesizer, Qwen trial

### 2026-03-25 | Research mode
- Shipped 4 research providers: Perplexity sonar-pro, o4-mini-deep-research, o3-deep-research, Gemini+Search
- Progressive Rich display, file cache (7-day TTL), result merger + LLM summarizer
- `--deep` flag for o3-deep (45 min, $10+); `--no-cache` bypass
- 35 new research unit tests

### 2026-03-22 | Mode system (pick/ideas/judge)
- Four debate modes with per-mode prompts and persona directives
- Auto-detection via cheap LLM call with 5s interactive confirm
- `-M` short flag (was `-m`, conflicted with `python -m`)
- 37 new mode unit tests

### 2026-03-20 | Default panel update + prompt upgrades
- Default panel: Claude + Gemini + OpenAI (was Claude + Gemini + DeepSeek)
- Round 1: structured decision framework; Round 2: steelmanning + hidden assumptions
- Synthesis: argument quality weighting + blind spot detection
- Fixed Gemini event loop crash (fresh `genai.Client()` per call)

### 2026-03-15 | Phase 1 foundation
- Multi-model debate pipeline: Claude, Gemini, GPT, Grok, DeepSeek
- Panel system, persona injection, blind voting (Round 2 anonymization)
- Non-participating synthesizer selection
- Inbox batch mode with frontmatter overrides
- Health checks at startup; cost tracking per debate
- 72 tests; CHANGELOG v1.0.0
