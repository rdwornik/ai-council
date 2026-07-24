---
last_reviewed: 2026-07-24
status: active
owner: Rob
---

# Architecture — `ai-council`

> Living document. Updated after structural changes.
> Last updated: `2026-07-23` (`pre-handoff claim-vs-reality reconciliation: codemap + responsibilities + layer tables gain boost/crux_check; config path corrected to repo-root config/; pre-commit roster reconciled to the live 12 hook ids; research model names reconciled to settings.yaml; invariants 1/5 reworded to match code (data shapes incl. enum/protocol; synthesizer fallback documented); local ADR span 01…14 — same-day architect repair pass: layer-edge set completed + the cli->boost interface->core edge named as an OPEN CASE (unruled, belongs with the #69/P2 wiring arc); invariants 2/3 marked target-vs-state (#92) / known-exception (#99); boost input stage added to Data Flow (separate invocation, P2 wiring pinned by the T8 strict xfail) and its council-brief-*.md landing recorded in Folder Governance; ADR-13 restored to the Governing-ADRs roster (the CLAUDE.md sibling was repaired a day earlier — live #97 rule-4 instance); stale 'pending #2' clause resolved (amendment verified landed, e3bdcc8/a854bd3)`)

## Purpose [CORE]

`ai-council` is a multi-model AI debate and research CLI with four operating modes (pick, ideas, judge, research). The operating modes are the *decide* half of a boost→decide chain: `council boost` is the *input* half that precedes them, turning a raw question from a methodology-naive caller into a type-classified boosted brief (decision / research / hybrid) before the debate runs (ADR-11 amendment 2026-07-22). It coordinates a configurable panel of AI models (Anthropic, OpenAI, Google, xAI, DeepSeek) through structured rounds of parallel debate and blind-vote critique, synthesizes a final verdict from a non-participating synthesizer, and routes transcripts to both its operational archive and target project directories. Within the `Dev/` ecosystem it is the primary mechanism for producing Architecture Decision Records: Council debate produces a verdict; that verdict is authored into an ADR that is ratified and distributed by `.dev-knowledge` to all downstream repos.

---

## Codemap [CORE]

> **Codemap form.** ai-council maintains this block **by hand** in the codemap tool's compact-text shape (ADR-51). The hub codemap CLI cannot generate it: ai-council has a flat single-package layout with no `tach.toml`, so `codemap generate` degenerates to a 2-orphan-module stub (#262 gap-note). The tables below show the module dependency structure; per-module responsibilities follow further down.

<!-- CODEMAP:START -->
<!-- HAND-AUTHORED compact-text codemap (ADR-25-style manual block, ADR-51 form); NOT generator-managed.
     The hub codemap CLI cannot regenerate this: ai-council is a flat single-package layout with no
     tach.toml, so `codemap generate` yields a degenerate 2-orphan stub (providers, research). Edit by
     hand when modules/edges change; `codemap check` will always report a diff here — that is expected
     and is not wired into any gate (#262 gap-note). -->

Modules (source root: `src/ai_council/`; layer per the layer model below):

| module | layer | path |
|---|---|---|
| cli | interface | src/ai_council/cli.py |
| inbox | interface | src/ai_council/inbox.py |
| orchestrator | orchestration | src/ai_council/orchestrator.py |
| runner | orchestration | src/ai_council/runner.py |
| doctor | orchestration | src/ai_council/doctor.py |
| boost | core | src/ai_council/boost.py |
| crux_check | core | src/ai_council/crux_check.py |
| debate | core | src/ai_council/debate.py |
| synthesis | core | src/ai_council/synthesis.py |
| mode_detector | core | src/ai_council/mode_detector.py |
| seat_router | core | src/ai_council/seat_router.py |
| providers | core | src/ai_council/providers/ |
| research | core | src/ai_council/research/ |
| output | output | src/ai_council/output.py |
| routing | output | src/ai_council/routing.py |
| models | foundation | src/ai_council/models.py |
| metrics | foundation | src/ai_council/metrics.py |
| healthcheck | foundation | src/ai_council/healthcheck.py |
| policy | foundation | src/ai_council/policy.py |
| config | foundation | config/ (repo root — top-level package, not under src/) |

Dependencies (`from -> to`):
- cli -> orchestrator
- cli -> doctor
- cli -> boost
- inbox -> orchestrator
- orchestrator -> runner
- orchestrator -> seat_router
- orchestrator -> crux_check
- boost -> mode_detector
- boost -> providers
- boost -> config
- crux_check -> metrics
- crux_check -> models
- crux_check -> providers
- crux_check -> research
- crux_check -> config
- doctor -> healthcheck
- runner -> mode_detector
- runner -> healthcheck
- runner -> debate
- runner -> config
- debate -> providers
- debate -> synthesis
- debate -> models
- debate -> seat_router
- seat_router -> providers
- seat_router -> models
- seat_router -> config
- synthesis -> models
- synthesis -> output
- output -> routing
- research -> providers
<!-- CODEMAP:END -->

**Module responsibilities** (semantic complement to the structural graph):

| Module | Layer | Responsibility |
|--------|-------|----------------|
| `cli.py` | interface | CLI entry; `@click.group` with `run` + `boost` + `doctor` subcommands (`_DefaultGroup` routes bare `council "q"` → `run`); Click args → RunRequest; health-check gate; **one** output-destination resolver shared by `run` + `doctor` (`--output` > `--no-persist` > `AICOUNCIL_OUTPUT_DIR` > config); boundary handlers that turn a required-write failure into a clean non-zero exit |
| `inbox.py` | interface | File-based batch input; YAML frontmatter; archive |
| `boost.py` | core | `council boost` input stage (ADR-11 boost→decide chain): raw methodology-naive question → type-classified brief (decision/research/hybrid); deterministic scaffolding — brief body is caller text + fixed template constants only; gaps become advisory `[BOOST-GAP]` flags; never enumerates options/constraints the caller did not supply |
| `crux_check.py` | core | Bounded crux check between Round 1 and Round 2 (#18): one LLM call names the central empirical crux from the anonymized Round-1 block, headless retrieval checks it, one evidence artifact injected into Round-2 prompts; never raises into the debate (CruxStatus: grounded / no_empirical_crux / retrieval_unavailable) |
| `orchestrator.py` | orchestration | CouncilRunner; debate lifecycle coordinator |
| `runner.py` | orchestration | Panel selection; provider init; mode resolution |
| `doctor.py` | orchestration | `council doctor` liveness + config truth table (GREEN/YELLOW/RED over keys/seats/config); advisory-only (never blocks a run); writes `output/health/doctor-*.json` (#25/ADR-08 exit convention) |
| `debate.py` | core | Parallel debate rounds; blind-vote critique; retry |
| `synthesis.py` | core | Final synthesis; transcript assembly; DebateResult build |
| `mode_detector.py` | core | LLM-based question classification (pick/ideas/judge) |
| `seat_router.py` | core | CLI-seat admission gate + same-seat API fallback; one `SeatMetrics` per seat → the `seats[]` sidecar (ADR-12, #16) |
| `providers/` | core | 5 debate-provider implementations + CLI-subscription adapters: `base.py` (AIProvider ABC), `anthropic.py`, `openai_provider.py`, `gemini.py`, `xai.py`, `deepseek.py`, `cli_base.py` (`CliProvider` + `ClaudeCliProvider`/`CodexCliProvider` — subscription backend behind the ABC, ADR-12) |
| `research/` | core | Research-mode subsystem (isolated code path): `runner.py`, `provider.py` (ABC), `cache.py`, `merger.py`, `models.py`, `display.py`, `output.py`, `providers/` (5 research providers) |
| `output.py` | output | Rich console render + markdown archive write; **one** verify-or-raise routed-write helper (`_write_routed`) that decides required-vs-best-effort per destination, verifies the write landed, and either accumulates a `RoutingFailure` or raises `OutputRoutingError` — a required `--return-dir` miss is never silent (#35/#62) |
| `routing.py` | output | TargetResolver; secondary transcript routing (ADR-43) |
| `models.py` | foundation | Pure dataclasses; all shared data shapes; no logic |
| `metrics.py` | foundation | Token-count + cost tracking per provider call |
| `healthcheck.py` | foundation | Provider health checks; startup gate |
| `policy.py` | foundation | RunPolicy; retry backoff spec |
| `config/` | foundation | `config_loader.py` (type-safe YAML → dataclass; appConfig singleton) + `settings.yaml` (single config source of truth: models, prompts, personas) |

---

## Layer Boundaries & Invariants [CORE]

### Layer model

Four layers in dependency order (interface → orchestration → core → foundation). `output` is a cross-cutting concern that writes results produced by `core`.

<!-- HAND-AUTHORED compact-text layer-boundary map (ADR-25-style manual block); NOT generator-managed.
     No CODEMAP markers -> the codemap CLI never reached this block even before the flat-layout issue. -->

Layers (dependency order; `output` is cross-cutting, written by `core`):

| layer | modules |
|---|---|
| interface | cli, inbox |
| orchestration | orchestrator, runner, doctor |
| core | boost, crux_check, debate, synthesis, mode_detector, seat_router, providers, research |
| foundation | models, metrics, healthcheck, policy, config |
| output (cross-cutting) | output, routing |

Layer edges (`from -> to`) — the allowed set (**TARGET, not current state** — see the dated
current-state note below). Enumerated against the codemap above; an incomplete set cannot serve as
a gate, so every edge that exists *in the codemap* is listed here or is named as an open case below.
**The set is not widened to match reality** — rewording a boundary to match a defect launders the
defect (R1, 2026-07-24). The gap between this set and `cli.py`'s real import surface is recorded in
the current-state note, not legalised.

- interface -> orchestration
- orchestration -> orchestration (same-layer: `orchestrator -> runner`)
- orchestration -> core
- orchestration -> foundation (`doctor -> healthcheck`, `runner -> healthcheck`, `runner -> config`)
- core -> core (same-layer)
- core -> foundation
- core -> output (writes via)
- output -> output (same-layer: `output -> routing`)

**OPEN CASE — `interface -> core`, one instance: `cli -> boost`.** Introduced 2026-07-22 with the
`council boost` input stage. It is the only edge in the codemap that skips a layer, and it is not
in the allowed set above. Two candidate resolutions, unruled:

  (a) `boost` is misclassified. Its work — selecting a cheapest provider, coordinating
      classify → decompose → reformulate → emit — is the same shape as `runner` (panel selection,
      provider init, mode resolution), which is **orchestration**. Reclassifying `boost` to
      orchestration dissolves the violation with no code change.
  (b) `interface -> core` is legitimate for single-stage commands that have no lifecycle to
      orchestrate, and should be added to the allowed set with that scope stated.

Do not resolve this by silently adding the edge. The ruling belongs with the P2 wiring arc
(BACKLOG #69), because that arc is what decides whether boost gains an orchestration entry point.

**STATUS: TARGET, not current state (R1 ruling, 2026-07-24; BACKLOG #92 is the refactor that
closes the gap).** The allowed set above is the boundary `cli.py` is held *to*, not a description of
what it does today. Re-derived live from source via AST on 2026-07-24 (TYPE_CHECKING-guarded imports
excluded), `cli.py` has **14 real internal module edges**: **3 fall inside the allowed set**
(`interface -> orchestration`: `cli -> orchestrator`, `cli -> runner`, `cli -> doctor`), **1 is the
named open case** (`cli -> boost`, above), and **10 are unaccounted** — real edges outside the
allowed set, not yet ruled:

- `interface -> interface`: `cli -> inbox`
- `interface -> core`: `cli -> mode_detector`, `cli -> providers`, `cli -> research`
- `interface -> foundation`: `cli -> healthcheck`, `cli -> models`, `cli -> policy`, `cli -> config`
- `interface -> output`: `cli -> output`, `cli -> routing`

These 10 are the concrete shape of Invariant 2's target-vs-state gap (`cli.run()` still does provider
loading, health checks, mode resolution and research dispatch — #92). They are **not** added to the
allowed set: the set stays the target, and #92's decomposition is what shrinks `cli.py`'s real
surface toward it. Do not cite the allowed set as a description of today's `cli.py`. **Re-derive this
note after #92 lands.**

**Codemap-completeness gap (distinct from allowed-set conformance).** The Codemap block above lists
only 3 of these 14 `cli` edges (`orchestrator`, `doctor`, `boost`), so **11 real `cli` edges are
absent from the hand-maintained codemap** (the 10 unaccounted above plus `cli -> runner`, which is
allowed but unmapped). Validating the map against itself would pass an omitted illegal import;
catching that is **BACKLOG #97 rule 14 leg (b)** — every real `src/` inter-module import must appear
in the codemap (map completeness against source). The follow-up codemap edit rides the same
#97/#92 arc; it is **not** applied here (this pass is allowed-set-honesty only, per R1).

**Enforcement tool:** Convention + code review. No automated import-linter at current scale (no Tach).
This is the known weak point: the codemap and this layer map are both hand-maintained, `codemap check`
always reports a diff here and is wired to no gate (#262 gap-note), and `check.ps1` (pytest + mypy +
ruff, plus a non-blocking #97 claim-check) does not inspect imports. Layer conformance is therefore **unverified by construction** —
mechanisation is BACKLOG #97 rule 14, two legs: leg (a) every codemap edge falls inside the allowed
set above; leg (b) every real `src/` inter-module import appears in the codemap (map-vs-source, which
de-vacuates leg (a) — an illegal import the map omits would otherwise pass). The current-state note
above is the hand-run of leg (b) against `cli.py`.
**Config file:** N/A
**Where enforced:** Code review only, pending #97.

### Module-to-layer assignment

| Layer | Modules |
|-------|---------|
| `interface` | `cli.py`, `inbox.py` |
| `orchestration` | `orchestrator.py`, `runner.py`, `doctor.py` |
| `core` | `boost.py`, `crux_check.py`, `debate.py`, `synthesis.py`, `mode_detector.py`, `seat_router.py`, `providers/`, `research/` |
| `foundation` | `models.py`, `metrics.py`, `healthcheck.py`, `policy.py`, `config/` |
| `output` (cross-cutting) | `output.py`, `routing.py` |

Utility-exemption modules (importable from any layer): none *(TARGET — `cli.py` currently reaches
`foundation` directly (`models`/`policy`/`healthcheck`/`config`), per the current-state note above;
the goal is that no module is a universal utility)*.

### Invariants

1. `models.py` contains no behaviour — pure data shapes only (dataclasses, the `CruxStatus` enum, the `CruxChecker` protocol, status-string constants); the sole source of shared data shapes across all layers.
2. `cli.py` performs no business logic — it translates CLI arguments into a `RunRequest` and immediately delegates to `CouncilRunner`.
   **STATUS: TARGET, not current state (BACKLOG #92).** `cli.run()` is currently ~196 statements
   (cyclomatic complexity ~50) and performs provider loading, the health-check gate, mode resolution
   and research dispatch; the module-responsibility table above records three of these deliberately.
   The invariant is retained as the goal rather than reworded to match the code — #92 is the refactor
   that makes it true. Do not cite this invariant as a description of today's `cli.py`.
3. Config strings (model names, prompts, personas, cost rates) are defined solely in `config/settings.yaml` — none hard-coded in Python source.
   **Known exception, unresolved:** two research providers carry constructor fallback defaults
   (`research/providers/openai_deep_research.py` and `openai_mini_research.py`). Either the fallbacks
   are stripped, or this invariant is scoped to runtime-selected values; the choice is un-made and
   sits in the un-triaged proposal set (BACKLOG #99). Until then this invariant is true of
   runtime selection and false of constructor defaults.
4. `_anonymize_responses()` in `debate.py` is the sole implementation of blind voting (ADR-03); its shuffle order is part of the contract and must not be altered without an ADR.
5. The synthesizer is a non-participating observer — it receives the full transcript but does not vote in debate rounds, and must not be a member of the debating panel, except the documented last-resort fallback: when no non-panel provider is available, `pick_synthesizer()` returns a panel member flagged `is_participant=True` (see Key Design Decisions).
6. The research subsystem (`research/`) is an isolated code path — features added to interactive debate mode are not automatically available to research mode; they must be explicitly mirrored.
7. All mutable session state flows through `DebateResult` and `RunRequest` dataclasses — provider implementations are stateless between calls.

→ Related decisions: `docs/decisions/ADR-03-blind-voting.md`, `docs/decisions/ADR-04-mode-system.md`, `docs/decisions/ADR-05-research-integration.md`, `docs/decisions/ADR-07-dual-output-paths.md`

---

## Data Flow

**Input stage (separate invocation, ADR-11 amendment 2026-07-22).** `council boost` is a
distinct subcommand that runs *before* and *outside* the pipeline below:

```
raw methodology-naive question (arg or --file)
       |
boost.py: classify (decision / research / hybrid) via one cheap LLM call
       |
       +-- hybrid: decompose into <=3 linked sub-briefs (research leg may feed a decision leg);
       |           split points must be a contiguous span of the caller's own text, else the
       |           legs fall back to full text and the run exits 3 (degraded)
       |
reformulate: brief body = caller text + fixed template constants ONLY; information gaps become
             advisory `[BOOST-GAP]` annotations; no option or constraint the caller did not
             supply is ever introduced (ADR-11 amendment (e) / hub ADR-95 boundary)
       |
emit `council-brief-*.md` with frontmatter written via `frontmatter.dumps()`
       |
   >>> the CALLER then invokes `council --file <brief>` itself <<<
```

**Not yet wired in.** Phase P1 shipped `boost` as a standalone subcommand only. Wiring it into
`run --file` / `--inbox` is Phase P2 and is deliberately deferred so that the inbox/CLI parity
defect (BACKLOG #69) is closed as one atomic unit rather than inherited by a new surface. The
constraint is pinned mechanically by a strict `xfail` in `tests/test_boost.py` (T8), which will
error the suite the moment wiring lands without parity.

End-to-end debate pipeline:

```
1. Input arrives via CLI (`council "question"`) or inbox file
       |
2. mode_detector.py classifies the question (pick / ideas / judge / research)
       |
3. runner.py selects panel + resolves synthesizer; healthcheck.py gates startup
       |
4. debate.py runs Round 1: parallel provider calls; anonymizes + shuffles responses (ADR-03)
       |
5. debate.py runs critique rounds: each panellist sees anonymized responses from prior round
       |
6. synthesis.py calls non-participating synthesizer on full transcript; builds DebateResult
       |
7. output.py writes Rich console display + the transcript (`council-out-*.md`, with a
   human-readable verdict mirror), the machine-authoritative **verdict package**
   (`council-verdict-*.json`, DRAFT-INT-1/#26), the metrics sidecar (`*_metrics.json` with the
   `seats[]` + `synthesis` namespaced blocks), and `council-minority-*.md` when dissent → ai-council/output/
       |
8. routing.py optionally copies curated transcript to target project dir (ADR-43)
```

Research mode branches at step 2: `research/runner.py` → cache check → parallel provider calls → `research/merger.py` → `research/output.py`. If fewer than 3 providers succeed, run continues but exits with code 3 and an alarm banner (ADR-08). (Research emits the report only — no verdict package yet; that parity is BACKLOG #34.)

**CLI-subscription seats (ADR-12, #16):** when a panel seat is configured `backend: cli`, `seat_router.py` runs it via the `claude`/`codex` subscription CLI ($0 marginal) between transport and round entry (step 4), with a same-seat API fallback recorded in the `seats[]` sidecar; the synthesizer is always API. **`council doctor`** (#25) is a separate advisory pre-flight (keys/seats/config truth table) and is never on the debate path.

---

## Key Design Decisions

- **Panel system**: `determine_panel()` in `runner.py`; `--models` wins over `--full`/`--lite` wins over default. Full 5-model panel is the default; `--lite` uses 3-model panel; `--full` is a no-op kept for backward compat.
- **Blind voting**: `_anonymize_responses()` shuffles + labels as "Proposal A/B/C"; provider names hidden during critique rounds (ADR-03).
- **Non-participating synthesizer**: `pick_synthesizer()` picks a model outside the panel; default `openai` (ratified 2026-07-18; was `gemini`, ADR-01 amended); falls back with `is_participant=True` if none available. Always API-lane (synthesizer-never-CLI guard).
- **Config source of truth**: All model strings, timeouts, max_tokens, prompts, personas in `config/settings.yaml` — none hard-coded.
- **Graceful degradation**: Round 2+ all-fail → `DebateOutcome(degraded=True)` with partial rounds; round 1 all-fail → `RuntimeError`.
- **Research mode**: Separate code path — bypasses debate pipeline entirely; runs parallel providers via `asyncio.wait()` + progressive display; merges results; summarizes via LLM; file cache under `~/.ai-council/research_cache/` with 7-day TTL.
- **CLI-subscription backend (ADR-12, #16)**: a panel seat may run on a subscription CLI (`claude`/`codex`) instead of an API endpoint (`backend: cli` per model in `settings.yaml`); `seat_router.py` gates admission on a witnessed served identity and falls back to the same-seat API on any CLI failure, recording `seats[].fallback_events[]`. CLI calls are $0 marginal; the default backend stays `api` (the §5 flip is evidence-gated on the #27 parity run).
- **Verdict package (DRAFT-INT-1, #26)**: every debate run emits `council-verdict-<ts>-<mode>-<slug>.json`, a transcript-free decision record (decision, rationale, options, dissent pointer, panel/seats, verdict author, degradation) to every destination — the machine-authoritative deliverable a Lane-A caller consumes without reading the transcript. Research-path parity is pending (#34).

---

## Transcript Routing

Opt-in, per-invocation mirroring of debate transcripts to named target project directories (ADR-43).

**Two-layer model:**
- **Names** (e.g., `.dev-knowledge`) come from frontmatter or `--target-project` flag — dynamic per invocation.
- **Ecosystem root** (`dev_root`) declared once in `config/settings.yaml`; paths computed as `<dev_root>/<name>/docs/decisions/transcripts/`.

**Behavior:**
- Canonical `output/` always written first (hard requirement; failure is hard error).
- Mirror writes are best-effort: failure logs warning, canonical never affected.
- Unknown target name → `RoutingError` at parse time (before debate runs), listing known names.
- No `target-project` → canonical only, unchanged behavior.

**Key files:** `src/ai_council/routing.py` (TargetResolver + RoutingError), `config/settings.yaml` (dev_root + target_projects list).

---

## Debate Modes

| Mode | Aliases | Default rounds | Purpose |
|------|---------|---------------|---------|
| `pick` | `p`, `pick`, `d`, `decide` | 2 | Choose between options **(default)** |
| `ideas` | `i`, `ideas` | 1 | Brainstorm; surface unknowns and divergent ideas |
| `judge` | `j`, `judge` | 2 | Evaluate a proposal or claim; get a verdict |
| `research` | `r`, `research` | — | Multi-source web research with citations |

- `pick` uses `prompts.initial`/`prompts.critique`/`prompts.synthesis` from `settings.yaml` (backward compat).
- `ideas`/`judge` use per-mode blocks in the `modes:` section of `settings.yaml`.
- Mode auto-detected from question text via cheap LLM call (5s interactive confirm); resolved by `resolve_mode()` in `config_loader.py`.
- `research` routes to `src/ai_council/research/runner.py:run_research()` before the debate pipeline.

---

## Research Providers

| Provider | Env var | Default | `--deep` only | Notes |
|----------|---------|---------|--------------|-------|
| `perplexity` | `PERPLEXITY_API_KEY` | yes | no | sonar-pro; OpenAI-compatible |
| `grok` | `XAI_API_KEY` | yes | no | grok-4.20-0309-reasoning; Responses API; unique X/Twitter signal |
| `openai_mini` | `OPENAI_API_KEY` | yes | no | gpt-5.4-mini + web_search (Responses API; migrated off o4-mini-deep-research 2026-05-18) |
| `gemini` | `GEMINI_API_KEY` | yes | no | Interactions API; autonomous agent; ~5-20 min |
| `openai_deep` | `OPENAI_API_KEY` | no | yes | gpt-5.5 + web_search, reasoning=high (Responses API; migrated off o3-deep-research 2026-05-18); ~45 min timeout |

Missing API keys are silently skipped — remaining providers still run.

---

## Folder Governance

| Folder | Contents |
|--------|---------|
| `src/` | All Python source code |
| `tests/` | All tests |
| `config/` | `settings.yaml` and `config_loader.py` |
| `scripts/` | `check.ps1` and utility scripts |
| `protocols/` | Outward-facing invocation specs (SCREAMING_SNAKE): `COUNCIL_QUESTION_GUIDE.md`, `SYNTHESIS_QUALITY_RUBRIC.md` — ai-council's delegation surface, mirroring the hub's `protocols/` (ADR-09, local) |
| `docs/` | `decisions/` (ADRs + `transcripts/`), `audits/` (reports; pre-ADR-34 in `audits/archive/legacy/`), `archive/` — ADR-60 child-repo taxonomy (no `handoffs/`; those centralize in `.dev-knowledge`). Invocation specs live in `protocols/`, not here (ADR-09) |
| `output/` | Gitignored; debate transcripts, `council-verdict-*.json` verdict packages (#26), `*_metrics.json` sidecars (`seats[]`/`synthesis`), research reports, `council-minority-*` dissent artifacts (#15), `output/health/doctor-*.json` records (#25), and `council-brief-*.md` boosted briefs from `council boost` (overridable per-invocation with `--out-dir`; sub-brief legs are suffixed `-1-research` / `-2-decision`, collisions with `-v2`/`-v3`). Canonical write; `--return-dir` additionally routes a copy (ADR-10, #13) |
| `council_inbox/` | Gitignored; drop `.md` files for batch processing |
| `LESSONS.md` | Repo-local lessons (append-only; at repo root) |

Do not create files outside these directories without updating this section.

---

## Inbox File Detection

`--inbox` processes files from two sources. Filename conventions differ — this is authoritative:

**1. `council_inbox/*.md` (primary inbox)** — `scan_inbox()` in `src/ai_council/inbox.py`
- Any `.md` file is picked up. No filename token or frontmatter required.
- Directory configured at `inbox.dir` in `settings.yaml` (default `./council_inbox`).
- Processed oldest-first by mtime.

**2. `~/Downloads/*.md` (opt-in pre-scan)** — `scan_downloads_folder()` in `src/ai_council/inbox.py`
- Picked up if **either**: (a) filename stem contains `council` (case-insensitive); **or** (b) YAML frontmatter contains any key in `inbox.council_frontmatter_keys` (default: `mode`, `rounds`, `models`, `synthesizer`, `full`, `target-project`).
- Files matching neither condition are silently ignored.
- Malformed YAML frontmatter is logged; file is skipped unless it already qualified by filename.
- Toggle via `inbox.scan_downloads`; directory via `inbox.downloads_dir`.

**Common to both:** Processed files move to `council_inbox/archive/` with `YYYY-MM-DDTHHMM_` prefix (or `FAILED_<timestamp>_` on failure). Prefixes are stripped from the slug used for output filenames via `clean_slug()`.

---

## Key conventions

- **Naming.** snake_case Python; kebab-case markdown; `ADR-NN-topic.md` for decisions (ADR-34); Council CLI output `council-out-YYYYMMDD-HHMMSS-topic.md`; ALL-CAPS for top-level governance markdown.
- **Append-only files.** `LESSONS.md` — never edit old entries (ADR-29). `JOURNAL.md` — newest-first prepend.
- **Immutable dated artifacts.** ADRs, transcripts, audits — supersede via a new file or in-file marker, never edit in place.
- **Config as source of truth.** Model strings, prompts, personas, timeouts, cost rates live solely in `config/settings.yaml` — none hard-coded (Invariant 3).
- **Layer discipline.** interface → orchestration → core → foundation; `output` is cross-cutting. `models.py` is logic-free; `cli.py` does no business logic *(target, not current state — see Invariant 2 / #92)*. Layer conformance is convention-enforced only; see § Layer model for the complete allowed edge set and the one open case.

---

## Authority and governance

`ai-council` is a **tool repo** governed by `.dev-knowledge` (Layer-2 binding authority, ADR-31). It owns its local tool-design ADRs (`docs/decisions/ADR-01…14`) and conforms to ecosystem ADRs (naming, file lifecycle, the seven-file canonical baseline ADR-38 A6).

- **Conformance:** verified out-of-band, read-only, by `.dev-knowledge/scripts/audit.py` against the canonical standard. `.dev-knowledge` never writes here (Layer-2 invariant, ADR-28).
- **Decision flow:** Council debate (this tool) → verdict → ADR authored + ratified by `.dev-knowledge` → distributed to downstream repos. Local ADRs cover only this tool's internal design.
- **Cross-domain split (ADR-67):** the process spec lives in `.dev-knowledge`; the `/council-question` template + gate + `council.return_dir` I/O are this repo's to implement (see `BACKLOG.md`).

---

## Validators and enforcement

- **`.\scripts\check.ps1`** — the pre-merge gate: `pytest` + `mypy` + `ruff`. Run before every merge (CLAUDE §5); not wired to pre-commit. A non-blocking #97 claim-vs-reality report (`scripts/validate_claims.py`) also runs as a section but does not gate.
- **`tests/`** — pytest unit + integration suites. Unit suite (no API keys): `pytest tests/ -m "not integration and not envcheck"`.
- **Pre-commit:** `normalize-headers` (dated-log header normalization in `LESSONS.md`/`JOURNAL.md`) · `floor-hash-verify` (`.claude/CLAUDE-FLOOR.md` vs its sha256 sidecar) · `canonical_freshness` (A2 `last_reviewed` gate; FAIL blocks the commit) · `validate-sealed-keys` (#67 — blocks a staged `SEALED-KEY*.json`; exact-path scoped override, never `--no-verify`) · `validate-docs-registry` (#68 — an unregistered new `docs/` directory blocks the commit; reads the registry from `docs/audits/README.md` at runtime and **fails CLOSED** as `GUARD MALFUNCTION`) · `validate-audit-casing` (ADR-101 R4 audit-filename casing) · `validate-backlog` (ADR-66 story-map structure) · `ruff` (consumer-owned lint gate, `astral-sh/ruff-pre-commit` mirror; re-activated 2026-07-12 by fleet ruling) · hub-sourced (`repo: ../.dev-knowledge`, pinned `rev`): `toc-freshness`/`toc-generate` (`protocols/COUNCIL_QUESTION_GUIDE.md`), `backlog-id-on-close` (requires `[#id]` in the commit message when a BACKLOG task is removed), and `block-ff-push` (pre-push; refuses a direct-to-main / FF push to `main`). Twelve hook ids total — this roster mirrors `.pre-commit-config.yaml`.
- **External conformance (read-only):** `.dev-knowledge/scripts/audit.py` — seven-file canonical baseline + structural spine (ADR-38 A6); manual `run`, no commit gating here.

---

## Governing ADRs

- **Local** (`docs/decisions/`): ADR-01 synthesizer selection · ADR-02 panel composition · ADR-03 blind voting · ADR-04 mode system · ADR-05 research integration · ADR-06 cost optimization · ADR-07 dual output paths (superseded by ADR-43) · ADR-08 research degradation alarm · ADR-09 protocols/ invocation surface · ADR-10 output routing · ADR-11 delegated invocation contract · ADR-12 provider backend engine · ADR-13 invocation-contract versioning · ADR-14 ADR lifecycle states.
- **Ecosystem** (`.dev-knowledge/docs/decisions/`): ADR-29 (append-only LESSONS) · ADR-34 (naming) · ADR-38 (namespace + A6 seven-file baseline) · ADR-42 (handoffs centralized) · ADR-43 (transcript routing) · ADR-51 (ARCHITECTURE convention) · ADR-53 (CLAUDE.md) · ADR-59 (visual pattern) · ADR-60 (docs taxonomy) · ADR-67 (Council process operationalization).

---

**Maintained by:** Rob
