# Night Batch — Unattended Empirical E2E Audit (2026-07-17)

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — #39/#40/#41 shipped, #42 struck; open: #43, #18, #19. _(Additive inventory stamp; body below unchanged.)_

**Class:** audits · **Status:** LIVE (night log, appended as legs complete) · **Author:** Claude Code (Opus 4.8, xhigh), on the architect's commission, operator pre-authorized
**Doctrine:** serial legs · hard stop conditions · **NO code edits · NO durable config edits · NO merges · NO ADR filings** · outputs = run artifacts + this one audits-class report
**Deliverable role:** this file is the morning report AND the intake for the next planning session.

> **Autonomous start (amendment):** the operator pre-authorized this batch. No interactive "go" was awaited. This Section 1 is the *echo-back-for-the-record* — the verbatim per-leg command matrix, models/effort, use-case list, verdict scale, morning-report path, and stop conditions. Execution began immediately after writing it. All stop conditions are in force and are the sole brake.

---

## 0. Morning TL;DR (read this first)

**The batch ran to completion. Zero stop conditions fired.** All 5 use cases ran; pre-flight GREEN; the config-override harness restored `config/settings.yaml` byte-identical after every run (repo code/config untouched).

- **PART 0 live trial — openai as synthesizer: PASS (4/4).** Every debate verdict was decisive, on-brief, and faithful to the transcript (correct per-panelist attribution, explicit strongest/weakest-argument calls, zero synthesis errors); in UC2 it *overrode* a leading design lean with a de-risked alternative. Corroborates the ruling; the EPI-1 pack stays the reversible instrument. **Durable settings swap deferred to your supervised micro-arc.**
- **CLI engine — claude + codex live end-to-end, ZERO fallbacks** across all 4 debates (claude-haiku via `modelUsage`, codex `gpt-5.6-sol` via stderr-banner). Subscription lane held; CLI calls billed $0. (Codex rode the `deepseek` seat-name tonight because openai was reserved as synthesizer — see §1.2; that indirection is itself a finding.)
- **Use-case verdicts:** UC1 Rama 1 **PASS**, UC2 Rama 3 **PASS**, UC3 DeepSeek **PASS**, UC4 currency **PASS**, UC5 research **DEGRADED** (retrieval fine, but gemini research-provider errored + deepseek summarizer truncation + **no verdict package = the #34 gap, live-confirmed**).
- **Aggregate council spend: ≈ $0.79** (gemini panel + openai synthesis + research pool; CLI seats $0). Quota: 7 claude-CLI + 7 codex-CLI subscription calls.

**Top findings needing your attention (evidence in §3/§4):**
1. **[hygiene, Med] Stray-`output/` root cause + guard gap CONFIRMED** — witness runs write the canonical `output/` unconditionally (no dry-run/scratch mode). You cleaned the stray transcripts but **`output/health/` still holds ~15 doctor records from today**. Candidate fix: a `--no-persist`/`--scratch` output mode.
2. **[hygiene, Med] `options_considered` is broken both ways** — empty on pick verdicts, populated-but-polluted on ideas (scoops endorsement lines). Affects the transcript-free ADR draft (§2.6).
3. **[hygiene/parity, Med] research path omits `--return-dir` and emits no verdict package** — the code-level root of #23 + #34.
4. **[metrics, Low] codex CLI token count reads 0 / claude reads input=9** — content is real, cost is $0, but token totals skew.
5. **[currency, the one immediate fix] BACKLOG `#31` is closed-but-still-listed (ADR-65 violation)** — strike it. `ARCHITECTURE.md` is the worst-stale doc (missing all of #16/#25/#26); CONTRACT §5 missing the verdict-package artifact.

**Nearest ready-slack next work:** **#22 + #23** (both unblocked by the G2 lift). **The Epic-B gate (#24/G3)** is what still blocks Rama 1 (#18) + Rama 3 (#19) — the two consolidation-brief survivors not yet delivered.

**Your three morning decisions:** (1) ratify the durable `defaults.synthesizer: gemini→openai` swap given the 4/4 evidence; (2) pick the next-wave items from §4.2 (recommend #22/#23 as ready-slack, + the `--no-persist` guard); (3) review + commit these 4 uncommitted deliverables (§6) and close the session / regenerate the handoff bundle.

---

## 1. Echo-back-for-the-record (the plan, verbatim)

### 1.1 Config-override harness (no durable config change)

`council --inbox` reads canonical `config/settings.yaml`; there is **no `--backend` flag and no `backend` frontmatter key**, so `backend: cli` can only be reached through settings.yaml. The harness is therefore a **transient swap** with guaranteed restoration (the established #16 pattern):

1. `Copy-Item config/settings.yaml → <scratch>/settings.yaml.pristine`, record SHA-256.
2. Swap in a mutated config that differs from pristine in exactly three places:
   - `models.claude`: `backend: cli`, `cli_command: claude`, `cli_model: claude-haiku-4-5-20251001`  → **CLI seat #1 (claude)**
   - `models.deepseek`: `backend: cli`, `cli_command: codex`, `cli_model: gpt-5.6-sol` → **CLI seat #2 (codex)**, hosted on the `deepseek` name (see 1.2)
   - `inbox.scan_downloads: false` → **Downloads safety** (see 1.3)
3. Run the one council command with a bounded wall-clock wait.
4. **`finally`**: restore pristine over settings.yaml, re-hash, assert byte-identical, assert `git diff --quiet config/settings.yaml`.

Restoration is wrapped in PowerShell `try/finally` **per invocation** — settings.yaml is pristine between every run and is restored even on error/kill. Nothing is committed; `settings.yaml` is byte-identical at session end.

### 1.2 The codex-seat / openai-synthesizer tension (a material decision, recorded)

The synthesizer is auto-excluded from the panel (`runner.exclude_synthesizer_from_panel`; `pick_synthesizer` prefers a non-panel model). The codex CLI adapter's canonical home is the `openai` model name (that is how #16 wired it). **Tonight PART 0 makes `openai` the synthesizer**, so `openai` cannot simultaneously be a codex *panel* seat. The provider registry (`PROVIDER_CLASSES` in `cli.py`) is a fixed set of six names — there is no first-class `codex` seat name available without a **code** edit (forbidden tonight).

**Decision (autonomous, recorded for morning review):** host the codex CLI seat on the **`deepseek`** model name (openai-compatible SDK for its API fallback; kept clear of the separate grok cost-lane work). The `seats[]` sidecar will read `seat="deepseek", cli.name="codex", actual_model="gpt-5.6-sol"` — a truthful record of "the deepseek-labelled seat was served by the codex CLI." This label indirection is itself a finding (feeds LEG 3 gap-map / theme [E5] naming): **the registry cannot first-class a `codex` seat.** Priority order applied: PART 0 (openai synthesizer — the gate event) > seat-label purity.

### 1.3 Downloads safety

`--inbox` also scans `~/Downloads` for council-keyed `.md` (default `scan_downloads: true`), and **Downloads currently holds council-named files** including `ai-council-architecture-consolidation-brief.md` and `2026-07-17-ai-council-p4-session-plan.md`. Left on, the batch would *process and archive-move* the operator's Downloads files. The harness sets `scan_downloads: false`, isolating every run to `council_inbox/` (operator-controlled). Verified clean between runs.

### 1.4 Per-leg command matrix (verbatim)

All commands run from repo root `C:\Users\1028120\Documents\Dev\ai-council`. Runner = `.venv\Scripts\council.exe` (foreign-caller shape). Each debate/research run = drop **one** brief to `council_inbox\<slug>.md`, then:

| Leg | Command (verbatim) | Swap? | Model / seats | Wall-clock kill |
|---|---|---|---|---|
| Pre-flight | `.venv\Scripts\council.exe doctor` | yes | doctor pings API backends | 180s |
| UC1 Rama1 (pick) | `.venv\Scripts\council.exe --inbox` | yes | claude(CLI-claude)+deepseek(CLI-codex)+gemini(API); synth **openai** | 900s |
| UC2 Rama3 (pick) | `.venv\Scripts\council.exe --inbox` | yes | same | 900s |
| UC3 #6 DeepSeek (pick) | `.venv\Scripts\council.exe --inbox` | yes | same | 900s |
| UC4 #17 currency (ideas) | `.venv\Scripts\council.exe --inbox` | yes | same, 1 round | 900s |
| UC5 #110 sycophancy (research) | `.venv\Scripts\council.exe --inbox` | **no (pristine)** | research pool (perplexity/grok/openai_mini/gemini) + deepseek summary; **non-deep** | 900s |

Frontmatter per debate brief: `mode`, `models: claude,deepseek,gemini`, `synthesizer: openai`, `rounds` (mode default). Research brief: `mode: research` only (no synthesizer/seats).

**Effort/model per driving leg:** LEG 1 driving + harness = me (Opus 4.8, xhigh). LEG 2 (hygiene) + LEG 3 (currency/gap-map) = read-only subagents. LEG 4 (optional EPI-1 judge) = subagent, skipped without penalty if quota tight.

### 1.5 Use-case list (real pending decisions / open backlog)

1. **UC1 — Rama 1** (consolidation brief §4): tool-grounded crux resolution, options (a) reuse research pool / (b) CLI agent panelist / (c) discrete crux-check step between rounds. Maps to #18 / DRAFT-EPI-3. **← amendment-2 consumption test applied here.**
2. **UC2 — Rama 3** (§4): static→active framing defense, options (a) false-consensus alarm / (b) debate-time re-derivation role / (c) keep static. Maps to #19 / DRAFT-EPI-4.
3. **UC3 — #6**: keep / replace / demote DeepSeek from the default full panel.
4. **UC4 — #17** (ideas): approaches to detect stale model configuration vs latest releases.
5. **UC5 — #110** (research): 2026 evidence on sycophantic convergence + blind-vote isolation in 2-round LLM debate; established mitigations.

### 1.6 Verdict scale (per run)

- **PASS** — pipeline completed; verdict package schema-valid & transcript-free-usable; synthesizer verdict is decisive, faithful to the transcript, and on-brief.
- **DEGRADED** — pipeline completed but with a material gap (missing artifact, seat fell back, thin/hedged verdict, or a known parity gap surfaced).
- **FAIL** — run did not complete, or the verdict is unusable/unfaithful, or a stop condition fired.

### 1.7 Stop conditions (in force)

- Any leg failing **twice** → stop, report, do not improvise.
- A debate hanging past its wall-clock budget → kill, record, continue serially.
- Any impulse to edit code or config **durably** → stop. (The transient restored swap is the sanctioned harness, not a durable edit.)
- `council doctor` RED at pre-flight → stop (amendment 1).

### 1.8 Explicitly NOT tonight

Durable synthesizer swap · #22/#23 micro-arcs · any ADR ratification · the handoff bundle (regenerated tomorrow) · any commit or merge. Deliverables are written to the working tree **uncommitted** for the operator's morning review.

---
## 2. LEG 1 — pre-flight + use-case debates (RESULT)

### 2.1 Pre-flight (amendment 1)

`council doctor` under the CLI swap → **verdict GREEN, exit 0** (no stop condition). All 6 KEYS present; all 6 SEATS ping OK (claude, claude-sonnet, deepseek, grok, openai, gemini-synthesizer); all CONFIG refs resolve. CLI F5 re-witness: **claude `2.1.212`, codex `codex-cli 0.144.5`** — identical to the #16 build witness. Harness round-trip byte-identical (`orig_sha == new_sha`, `restore_ok=True`, `git_clean_settings=True`) on every invocation. Downloads confirmed shielded (`scan_downloads: false`).

### 2.2 Per-use-case results (product-owner view)

Every debate ran the full pipeline: **inbox drop → `council --inbox --skip-health-check`** under the swap; panel `claude(CLI-claude)+deepseek(CLI-codex)+gemini(API)`; **synthesizer openai `gpt-5.4`** (non-participant). Zero seat fallbacks on any run. `--skip-health-check` used because doctor is the liveness pre-flight and it avoids the interactive `click.confirm` hang in an unattended batch.

| UC | Question (real decision) | Mode | What happened | Synthesizer verdict @ a glance | Cost | Verdict |
|----|--------------------------|------|---------------|--------------------------------|------|---------|
| **UC1** | Rama 1 / #18 — how to ground empirical cruxes mid-debate (a/b/c) | pick, 2rd | Full pipeline; claude-haiku + codex-gpt-5.6-sol + gemini-3.1-pro; non-unanimous → minority emitted; 228.7s | **Decisive + faithful**: "Adopt **(c)** bounded crux-check step between rounds, one canonical evidence artifact for all Round-2 prompts." Names each panelist's real contribution; calls strongest/weakest argument. Independently matches the DRAFT-EPI-3 design. | $0.1240 | **PASS** |
| **UC2** | Rama 3 / #19 — author-time → debate-time framing defense (a/b/c) | pick, 2rd | Full pipeline; non-unanimous → minority; 191.2s | **Decisive**: "Choose **(b)** debate-time framing-challenge ROLE, shadow-mode first." *Diverges* from the L-EPI alarm-first lean — a genuinely useful council input, de-risked via shadow mode. | $0.1156 | **PASS** |
| **UC3** | #6 — keep/replace/demote DeepSeek from default panel | pick, 2rd | Full pipeline; non-unanimous → minority; 185.8s | **Decisive**: "**DEMOTE** from default, keep opt-in + summarizer behind a reliability review." On-brief, reversible-framed. | $0.1088 | **PASS** |
| **UC4** | #17 — detect stale model pins (ideation) | ideas, 1rd | Full pipeline; unanimous → no minority; 57.0s | **Decisive next-step**: "Prototype a **doctor freshness check** with three layers." Ideas synthesis structure intact. | $0.0287 | **PASS** |
| **UC5** | #110 — sycophantic convergence + blind-vote isolation (2026 evidence) | research | Separate research path; perplexity/grok/openai_mini **ok** (34 sources), **gemini research provider errored**, **deepseek summarizer unavailable → truncation fallback** (ADR-08 graceful). **No verdict package** (research-path #34 gap, live-confirmed). 192.1s | N/A (research has no synthesizer). Report usable (34 sources, on-topic, cited) but summary is concatenated-truncated, not synthesized. | $0.4136 | **DEGRADED** |

**Where the outputs live** (all under `output/`, gitignored): `council-verdict-20260717_230406-pick-uc1-…json` / `_230852-pick-uc2` / `_231220-pick-uc3` / `_231341-ideas-uc4`; matching `council-out-*.md` transcripts + `*_metrics.json` (seats[] + cost) + `council-minority-*.md` (UC1-3); research report `council-out-20260717_231723-research-research-sycophantic-convergence-and-blind-vote-is.md`. Briefs archived to `council_inbox/archive/2026-07-17T23*`.

### 2.3 PART-0 live trial verdict (synthesizer gemini→openai)

**openai `gpt-5.4` as synthesizer: PASS across all 4 debates.** Every verdict was decisive (no fence-sitting), on-brief, and **faithful to the transcript** (each synthesis attributed positions to the correct panelists and called out the single strongest/weakest argument). UC2 shows it will *override* a leading design lean with a de-risked alternative rather than rubber-stamp. No synthesis errors (`error_class=none`), latency 11–45s. This is supporting evidence for the operator's ruling; the EPI-1 pack stays the reversible instrument.

### 2.4 Seat / backend evidence (CLI engine live)

All 4 debates: `claude` seat → `backend cli`, `cli=claude 2.1.212`, `identity_channel=modelUsage`, served `claude-haiku-4-5-20251001`; `deepseek` seat → `backend cli`, `cli=codex 0.144.5`, `identity_channel=stderr-banner`, served `gpt-5.6-sol`; `gemini` seat → `backend api`. **Zero `fallback_events` across every run** — the CLI subscription lane held end-to-end. CLI calls recorded `$0.00` (subscription lane). **Quota consumed:** 7 claude-CLI calls + 7 codex-CLI calls total.

### 2.5 Metrics-accounting findings (surfaced by the live runs)

- **F-M1 — codex CLI token count = 0.** Every `deepseek`(codex) call recorded `output_tokens: 0` though the content is real and substantive (UC1 synthesis rates it "strongest overall on system design"). The `CodexCliProvider._TOKENS_RE` (`tokens used[:\s]+…`) does not match codex `0.144.5`'s stderr token banner → `token_count=None`→0. Effect: `total_tokens` under-counts; **cost unaffected** (CLI = $0). Candidate fix: update the token regex to codex 0.144.5's actual banner.
- **F-M2 — claude CLI input_tokens ≈ 9.** Every claude call recorded `input_tokens: 9` for multi-paragraph prompts — `modelUsage.usage.input_tokens` under-reports prompt input (likely excludes cached/system portion). Cost unaffected (CLI = $0); token totals skewed.

### 2.6 Consumption test (amendment 2) — candidate ADR from the verdict JSON alone

Drafted a full candidate ADR for UC1 **from `council-verdict-…-uc1-….json` only** (transcript + minority deliberately unread). Result: **usable** — decision, rationale spine, panel, author, and dissent-existence are all present and faithful; a competent caller can produce a defensible `pick`-mode ADR from the JSON alone. **Fields that had to be worked around:** `options_considered.items` is empty (options recovered from the raw `question` field, not a synthesized field); no per-option vote tally; dissent is a one-line gist + a pointer (substance lives in the separate minority file); only Recommended-Decision + Argument-Quality sections are extracted (Consensus/Risks/Action-Items are not); no confidence field; `contract_version=null`. Full artifact + field-gap list: `docs/audits/2026-07-17-night-batch-candidate-adr-from-verdict-uc1.md` (raw input for #38).

### 2.7 LEG 1 cost tally

| Run | Cost |
|-----|------|
| doctor pre-flight | ~$0.00 (health pings) |
| UC1 Rama 1 | $0.1240 |
| UC2 Rama 3 | $0.1156 |
| UC3 DeepSeek | $0.1088 |
| UC4 currency (ideas) | $0.0287 |
| UC5 sycophancy (research) | $0.4136 |
| **LEG 1 total** | **≈ $0.79** |

Quota: 7 claude-CLI + 7 codex-CLI subscription calls (CLI seats billed $0). API dollars: gemini panel + openai synthesis + the research pool.

## 3. LEG 2 — artifact hygiene audit (read-only; cross-checked by two independent passes, consistent)

**Severity roll-up: 3 Med (H2-options, H3-research-routing, H4-stray-item guard gap), rest Low/PASS.**

### 3.1 Naming (Axis 1)
- **PASS** — tonight's 5 runs conform; each run shares exactly ONE `<ts>-<mode>-<slug>` stem across all its artifacts (`save_verdict_package` derives `run_id = transcript_path.stem`, `output.py:764`; minority uses `stem_base`, `orchestrator.py:188`). Per-run file counts: pick runs 4 each, ideas 3, research 1.
- **[Low] H1 — research double-prefix** `council-out-…-research-research-sycophantic-…`: `research/output.py:41` hardcodes the `research` mode token while the slug (from a query beginning "Research…") also starts with "research". The debate path uses the dynamic `result.mode` and doesn't double. Fix: strip a leading `research-` from the slug, or route research through the shared `_ts()`/mode formatter.
- **[Low, no action] Legacy corpus drift:** of 253 `output/` files, ~149 use the pre-`council-` scheme and ~16 `council-`prefixed May-2026 files have UPPERCASE/underscore slugs (e.g. `…-council-Q1-…`, `…-COUNCIL-BRIEF-…`). All historical, all gitignored — immutable local cruft, not a live violation (the `#8` timestamp-underscore is excluded, not re-reported).

### 3.2 Package schema (Axis 2)
- **PASS** — all 4 verdict JSONs valid with the exact DRAFT-INT-1 field set (`missing:[] extra:[]`); `contract_version:null` + `exit_semantics:0` on all four; `panel.seats` len 3; `requested==seated==[claude,deepseek,gemini]`, `dropped:[]`. Metrics sidecar `seats[]` + `synthesis` namespaces are additive top-level keys (ADR-12/L-CLI seam) — PASS.
- **[Med] H2 — `options_considered` is broken in BOTH directions.** Empty (`items:[]`, `heading:None`) on the 3 **pick** runs — pick-template alternatives sit under a heading none of `_OPTIONS_HEADING_MARKERS` (`output.py:555-561`) catch. But populated-**and-polluted** on the **ideas** run (uc4): the `"Top Tier"` marker matched and `_extracted_options` (`output.py:594-606`) scooped sub-bullets/endorsement lines as options — verbatim junk items `"Who endorsed it:** claude, deepseek, gemini"` and `"Provider-specific API/model listing adapters**"` (leaked `**`). Fix: add a pick-alternatives marker AND tighten `_extracted_options` to top-level bullets only + strip trailing `**`. (This is the deeper root of the §2.6 consumption-test gap.)
- **[Low] H2b — decision.value leaks internal `**`.** e.g. uc2 `"Choose (b): … ROLE**, … **bounded, …**"`. `_one_line` (`output.py:564-570`) strips only *wrapping* emphasis, not inline `**`. Cosmetic but it reaches the machine field.

### 3.3 Routing discipline (Axis 3)
- **PASS on containment** — NO writer drops a file outside the sanctioned set {canonical `output/`, `secondary_dir` (only if it exists), `return_dir` (ADR-10), `target_paths` (ADR-43)}. Debate writers all funnel through `_write_routed` (`output.py:143-198`); `save_verdict_package` hard-fails via `OutputRoutingError` on a required-`return_dir` miss (R4).
- **[Med] H3 — the research path re-implements routing and OMITS `return_dir`.** `save_research_to_file` (`research/output.py:29-114`) does not call `_write_routed` and has **no `return_dir` param**; `run_research` (`research/runner.py:133-143`) never threads it. So a research `--return-dir` commission silently doesn't deliver ADR-10. This is exactly BACKLOG **#23** (and overlaps #34) — now with the code-level root cause.
- **[Low] H3b — two canonical-only un-routed writers:** `_save_metrics_json` (`output.py:785-830`) writes the sidecar only to the canonical dir (a `--return-dir` caller gets transcript+verdict+minority but **not** the metrics sidecar — overlaps #35); `doctor.py::write_record` writes `output/health/` unconditionally. Both stay inside `output/` — no escape.

### 3.4 Root-cause of the stray `output/` item + guard gap (Axis 4) — the headline
**Root cause (best-evidenced): witness-run detritus from today's #16/#26 development, written into the unconditional canonical `output/`.** Every witnessed dev run drives `save_to_file`/`save_verdict_package`/`save_minority_report`, each of which **always** writes the canonical copy (`output.py:161-164`, `primary.write_text(...)` — no dry-run branch). Yet `output/` has **no `council-out-*` dated 20260717 before tonight's 23:04** (newest prior is `20260629_125409`); because `output/` is gitignored, deleting those witness transcripts left no git trace — consistent with "the operator cleaned a stray item today." **What the operator MISSED and is still there: `output/health/` holds 14 `doctor-*.json` records dated 2026-07-17 (14:59→22:56)** + `doctor-latest.json` (my pre-flight added a 15th at 22:56) — un-cleaned witness detritus from the #25/#16 doctor runs.

**Guard gap: CONFIRMED (YES).** `grep -niE "dry.run|ephemeral|no.?persist|scratch" src/ai_council/cli.py` → nothing. There is no scratch/dry-run/no-persist output mode anywhere; both `_write_routed` and `doctor.py::write_record` write the canonical tree unconditionally, so dev/witness/test runs cannot avoid polluting `output/`, and cleanup is manual and easy to miss (as `health/` proves). **Fix direction:** add a `--no-persist`/`--scratch` (or `AICOUNCIL_OUTPUT_DIR` env override) for witness runs, and auto-prune / explicitly document the `output/health/` records. *(This is a strong candidate next-wave item — see gap-map.)*

### 3.5 `.gitignore` (Axis 5)
- **PASS** — `.gitignore:37` `output/` covers every run artifact incl. `output/health/` + `output/research-questions/`; no `council-*` file is tracked (`git ls-files` empty for them). epi1-archaeology seal lines present (`.gitignore:55-61`) with the blind-seal rationale. Nothing council-relevant is mis-ignored in either direction.

## 4. LEG 3 — docs/state currency audit + gap-map (proposals only, NO edits made)

### 4.1 Currency findings vs today's 7 merges (#16/#25/#26/#28/#29/#30/#31)

**A1 — `ARCHITECTURE.md`: STALE (worst offender).** `last_reviewed: 2026-07-11` (line 2) predates all three structural merges; grep for `verdict|seats[]|backend|doctor|cli_base|seat_router|CliProvider|@click.group` = **zero hits**. Proposed updates (additive/in-place, nothing deleted):
- Codemap module table (L33–48): **add `doctor.py` and `seat_router.py`** rows (both shipped today).
- `providers/` line 78: **add `cli_base.py`** (`CliProvider`/`ClaudeCliProvider`/`CodexCliProvider`, the CLI-subscription backend behind the ABC).
- Dependency edges (L50–64): add `orchestrator`/`debate` → `seat_router`, `cli` → `doctor`.
- `cli.py` row (L71): rewrite — now a `@click.group` with `run` + `doctor` (`_DefaultGroup` routes bare `council "q"` → `run`).
- Data Flow (L145–167): step 7 must add the **verdict-package write** (`council-verdict-*.json`) + the `seats[]`/`synthesis` metrics-sidecar namespacing.
- Key Design Decisions (L171–179): add CLI-subscription backend (ADR-12) + verdict package (DRAFT-INT-1/#26).
- Folder Governance (L240): `output/` list omits `council-verdict-*.json` and `output/health/doctor-*.json` — add both.
- Governing ADRs (L298): **stops at ADR-10 — extend to ADR-11/12/14**.
- Re-stamp `last_reviewed` (L2) + "Last updated" (L10) after updating.

**A2 — `CLAUDE.md`: largely CURRENT, one date bug.** §11 ADR roster through ADR-14 is present + matches the README index (verified). **BUT footer L246 `**Last updated:** 2026-07-13` contradicts frontmatter `last_reviewed: 2026-07-17` + the v2.10 history entry** → update footer to 2026-07-17. (LOW/pre-existing: §11 L223 calls `council.return_dir` "not yet built" — the I/O half shipped via #13/ADR-10; only the #9 template+gate remains.) `council doctor`'s absence from §7 is correct (it's a CLI subcommand, not a slash command).

**A3 — `BACKLOG.md`: one ADR-65 VIOLATION (the single immediate hygiene fix).** **`#31` (GOV-1 execution) is CLOSED but STILL LISTED** as an open task at L120 under `[S12]`. It was closed today (merge `34be9aa`, "feature-work pause LIFTED"; done-when met, VISION/CONTRIBUTING reconciled) — the close-commit cited `[#1]` (absorbed) but **never struck its own #31 line**, and the 2026-07-17 grooming-log entry has no "#31 struck" note. PROPOSE: strike the #31 line + add a grooming strike-note; retain the `[S12]` story header. All other closed items (#16/#25/#26/#28/#29/#30) correctly struck; all residuals (#32/#33/#34/#35, [S13]/#36–#38) present + correctly stated.

**A4 — `protocols/COUNCIL_INVOCATION_CONTRACT.md`: §7 accurate, §5 STALE.** §7 Known deviations **both still hold verbatim** — deviation 1 (`--file` frontmatter) = #22 OPEN, deviation 2 (research `--return-dir`) = #23 OPEN; neither shipped today (deviation 2 re-confirmed empirically tonight). §5 Artifacts (L74–85) **omits `council-verdict-<ts>-<mode>-<slug>.json`** (DRAFT-INT-1/#26) → add it, flagged **debate-path-only** (research parity = the open #34 gap, confirmed tonight). Minor: note the `seats[]`/`synthesis` namespacing on `_metrics.json`; §8 walkthrough could name `council doctor` (step 3) + the verdict JSON (step 6).

**A5 — `VISION.md` / `CONTRIBUTING.md`: CURRENT** (GOV-1 already reconciled + re-stamped both to 2026-07-17). Optional VISION nit: Scope still frames the 5 providers as "all API" without the CLI-subscription option.

### 4.2 GAP-MAP — what remains for plan-of-record + consolidation-brief DONE

**Plan-of-record phase status:** P0 done · **P1 (#31) done today** (line-strike pending, A3) · P2 (#24) OPEN · **P3 (#28/#29/#30) done today** · **P4 (#25/#16/#26) COMPLETE today** · P5 OPEN · P6 OPEN.

**Consolidation-brief Rama survivors:** **Rama 2 (CliProvider) = #16 DELIVERED today** (tonight's E2E: claude+codex seats end-to-end, zero fallbacks — healthy); **Rama 4 (minority first-class) = DELIVERED** (#15, further surfaced by #26). **Rama 1 (#18) + Rama 3 (#19) = OPEN, baseline-gated on G3/#24.**
**§6 forks:** 6.1 interaction model → RESOLVED (batch retained, ADR-11); 6.2 cost model → substantially resolved via ADR-12 (default-flip pending #27); **6.3 scope boundary → NOT formally adjudicated** (implicitly held at "pure governance") — the one fork worth an explicit ruling.

**Candidate next-wave items (size: S ≤1 session · M ~1–2 · L multi):**

| # | Item | Closes | Size | Gate |
|---|------|--------|------|------|
| **#24** EPI-1 archaeology | **P2 = the G3/Epic-B event** | M (zero code) | pause-independent; un-gates #18/#19/#9/#2/ADR-13; tonight's decisive+faithful openai synthesis is a fresh data point |
| **#22** `--file` frontmatter | P6 deviation-1 | S | **UNBLOCKED (ready slack)** — structural basis fell out of #25's shipped `@click.group` |
| **#23** research `--return-dir` | P6 deviation-2 | S | **UNBLOCKED (ready slack)** — re-confirmed open tonight |
| P6 completion verify | P6 backstop | S | after #22+#23: empty CONTRACT §7 + DRAFT-INT-2 `1.0` stamp |
| **#34** research verdict-package parity | Rama-4 lane-parity / INT | M | **empirically confirmed open tonight** (research emits no verdict package) |
| **#35** broad R4 fail-loud return-dir | INT R4 | S | extends #26's `OutputRoutingError` |
| **#33** terra pass-3 (#26) | S10 residual | S | **date-gated on/after 2026-07-23** (codex credits) |
| **#32** doctor-v2 CLI auth-lane | S11 (#16 waived) | S | tonight's zero-fallback E2E does NOT verify the auth *lane* (subscription vs stored API key) — still the live gap |
| **#27** CLI-4 parity → default-flip | P5; finalizes §6.2 | M | prereq #16 ✓; tonight's zero-fallback CLI E2E de-risks it |
| **#18** Rama 1 crux resolution | P5; Rama 1 (keystone) | M | **baseline-gated G3/#24** |
| **#19** Rama 3 framing defense | P5; Rama 3 | L | **baseline-gated G3/#24**; runtime-coupled |
| **#9** `/council-question` template+gate | P5; ADR-67 | L | deferred until baseline settles + G3; reconcile w/ #36 |
| **#2/#3/#4** synthesizer Branch A/B | E2 | M/S | **BLOCKED on #24** |
| **#36/#37/#38** caller-side advisor | E1-S13 | M/S/S | #38 enabled by #26; #36 reconcile w/ #9 |
| **#10/#11** rubric + handshake data | E6 | S | #10 aided by tonight's faithful-synthesis data point |

**Bottom line:** **Plan-of-record** is complete through P4; DONE needs P2 (#24)→G3, then P5 (#27 + G3-gated #18/#19/#9) + P6 (#22/#23 + the separate hub arc). **Consolidation-brief** DONE needs Rama 1 (#18) + Rama 3 (#19) — both behind the #24/G3 Epic-B gate — plus a formal/explicitly-deferred §6.3 ruling; Rama 2 (#16) + Rama 4 delivered, §6.2 residual = #27. **Nearest ready-slack: #22 + #23** (both unblocked). **Immediate hygiene: strike the closed-but-listed #31 (A3).**

## 5. LEG 4 — LLM-judge second opinion over the EPI-1 pack: SKIPPED (with rationale)

**Decision: skipped** — and not merely on the "skip if quota is tight" clause (quota was not tight: ~$0.79 API + 14 CLI-subscription calls all night). The stronger reason is **non-duplication + moot-by-ruling**:

- A **5-way blind Sonnet LLM-judge second-opinion pass already exists** from the 2026-07-16 EPI-1 prep: `docs/audits/2026-07-17-epi1-archaeology-SECOND-OPINION-judge.md` (gitignored), result **gemini ≈ openai (95% each)**, explicitly segregated as second-opinion, never the verdict.
- **PART 0 resolved the synthesizer question by operator authority tonight** (gemini → openai), with the EPI-1 40-item pack **retained UNSCORED as the reversible instrument**. A fresh LLM-judge pass would not change the ruling and would re-do existing work.

Available to re-run on request if the operator wants a fresh second-opinion data point against the retained pack; it stays second-opinion-only per the sealed method (operator scores blind; LLM-judge never the verdict).

## 6. Deliverables produced tonight (all UNCOMMITTED — for operator morning review)

Per doctrine (NO merges, NO commits): everything below is in the working tree awaiting the operator's review. `config/settings.yaml`, `src/`, and all tracked code are **byte-identical** to session start (`git diff` empty).

**Tracked-eligible deliverables (untracked in working tree):**
| File | What it is |
|------|-----------|
| `docs/audits/2026-07-17-night-batch-empirical-e2e-audit.md` | **This morning report** (also the intake for the next planning session) |
| `docs/audits/2026-07-17-synthesizer-ruling-gemini-to-openai.md` | PART 0 operator ruling record (audits-class; Epic-B gate event) |
| `docs/audits/2026-07-17-night-batch-candidate-adr-from-verdict-uc1.md` | Amendment-2 consumption test: candidate ADR from the UC1 verdict JSON alone + field-gap list |
| `docs/intake/2026-07-17-hub-feedback-session-close-gate.md` | ADDENDUM filing: NEEDS-RULING hub note (Stop-gate handoff block + consumer hub-write guard) |

**Run artifacts (gitignored, under `output/`):** 4 debate transcripts + 4 verdict packages + 4 metrics/seats sidecars + 3 minority reports + 1 research report; doctor health record under `output/health/`. 5 briefs archived under `council_inbox/archive/2026-07-17T23*` (gitignored).

**Not created / not touched (doctrine):** no code edit, no durable config edit (settings.yaml restored + verified byte-identical after every run), no commit, no merge, no ADR ratified, no handoff bundle (regenerated tomorrow by design), no `JOURNAL.md` entry (session-close is the operator's morning event).

## 7. Session-end state (for the operator)

- **Working tree:** dirty **by design** — 4 untracked deliverables above; zero tracked-file modifications. This is the intended night-batch terminal state (uncommitted for morning review). The ADR-85 Stop-gate will flag the dirty tree; that is expected — resolve at the morning session (review → commit/merge the deliverables, or `/override` for this HEAD if closing without them). This exact friction is what the hub-feedback filing (§6) asks the hub to make mechanical.
- **Harness safety:** the config-override harness swapped `backend: cli` (claude+codex-on-deepseek) + `scan_downloads: false` transiently per run, restoring `config/settings.yaml` from a byte-exact pristine copy with SHA-256 verification in a `finally` block every time (`restore_ok=True`, `git_clean_settings=True` on all 6 invocations). Downloads never touched.
- **Stop conditions:** none fired. No leg failed even once; no debate hung; no impulse to edit code/config durably; doctor GREEN at pre-flight.

<!-- LEG 2 / LEG 3 findings to be inserted at §3 / §4 on background-audit completion. -->


