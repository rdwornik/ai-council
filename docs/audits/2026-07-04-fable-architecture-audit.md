# ai-council — Architecture Audit, Gap Analysis vs Consolidation Brief, ADR Drafts

> **Deployment-Status (2026-07-18 inventory):** PARTIAL — ADR-11/12 ratified (`c11fb42`), ADR-13 ratified this session; D2 #22/#23, D6 #16; open: #27, #9, #18, #34, #35, #43. _(Additive inventory stamp; body below unchanged.)_

**Date:** 2026-07-04 · **Mode:** PLAN — review artifact only, zero implementation, zero commits
**Status:** Draft — ADR seed (pre-decision). Audit ran in the `fable-audit` worktree under MODE PLAN (no repo changes); persisted to `docs/audits/` the same day on operator instruction. The ADR-11/12/13 texts inside are **drafts** — nothing is ratified by this document; `docs/decisions/` is unchanged.
**Auditor:** Fable 5 (single-model audit + synthesis, per operator instruction)
**Evidence base:** live worktree `fable-audit` @ `bda4fff` (main), hub `.dev-knowledge` live files, `C:\Users\1028120\Downloads\ai-council-architecture-consolidation-brief.md` (the TARGET input). Every current-state claim below is cited to a live file:line read during this audit — not to the brief, not to memory.

---

## Context

ai-council is the ecosystem's multi-model deliberation engine and intended "window on the world" — the surface through which other repos commission architectural decisions. The operator's consolidation brief (Layer-1 analyst output) proposes a direction; Faza 0 subsequently shipped a `protocols/` surface, ADR-09/10, return-dir output routing, and a first-class minority report — none of which the brief reflects. This audit verifies the live state, measures the gap to the target (real delegation over API-or-CLI backends with a sane cost model), makes the architectural calls, and drafts ADRs from ADR-11 — all for operator + primary-chat-architect review before anything is implemented.

---

## 0. Corrections to the prompt/brief premises (all verified live)

These change how the rest of the audit reads. Verified personally, not delegated.

1. **The synthesizer-identity premise in the audit prompt is FALSE for the live code.** The claim was: "`pick_synthesizer()` excludes any panelist, so on full-panel runs the de-facto synthesizer is often openai, not Gemini." In reality `CouncilRunner.run` calls `exclude_synthesizer_from_panel()` **before** `pick_synthesizer()` (`orchestrator.py:61-63` then `:72-74`). On a default run (full panel `[claude, gemini, deepseek, openai, grok]`, preferred `gemini`), gemini is **evicted from the panel first** (`runner.py:44-56`), then selected as a non-participating synthesizer (`runner.py:59-75`). **The declared default IS real: gemini synthesizes, as ADR-01 says.** `git log -S` shows the eviction step has existed since 2026-02-25/03-29 (`69c2f5d`, `0e21fed`, `95e4035`) — long before the June-16 audit.
2. **The real synthesizer finding is different and twofold:** (a) the "full 5-model panel" default (ADR-02, ARCHITECTURE.md:188) **de-facto debates with 4 models** — gemini never debates on defaults because it is pulled out to synthesize; (b) the de-facto-openai phenomenon exists, but its cause is **documentation**: every frontmatter example in `protocols/COUNCIL_QUESTION_GUIDE.md` hardcodes `synthesizer: openai` + `models: claude,gemini,deepseek,grok` (`:46-48`, `:382-386`, `:511-515`), and the inbox loop honors frontmatter `synthesizer:` (`cli.py:478-482`). Operators following the guide's examples get openai-authored verdicts. The guide's own rule text (`:481` — "it picks a non-participant automatically") mis-describes the mechanism (it evicts the synthesizer from the panel; it does not pick a different synthesizer).
3. **The June-16 audit never claimed a synthesizer-identity bug** — it scores "Non-participating synthesizer ✅" citing both functions (`docs/audits/2026-06-16-...md:62`). It does contain its own staleness (`:56` says default panel = 3, `--full` = 5 — wrong since the ADR-02 2026-05-11 revision made 5 the default).
4. **The Epic C CLI recon cited in the prompt does not exist in the repo.** Grep for `claude -p`, `codex exec`, `gemini -p`, `--output-format` etc. across the worktree: only hit is BACKLOG #16's story text, which names zero flags. The flags in the operator prompt (`claude -p --output-format json --tools ""`; `codex exec --sandbox read-only --json`; `gemini -p -o json`) are treated below as **operator-supplied input, unverified, to be re-verified at build time**.
5. **Brief §2 staleness confirmed** (as the prompt warned): dissent IS now a first-class artifact (#15 shipped, `output.py:403-447`); output routing shipped (#13, ADR-10); `protocols/` surface exists (ADR-09); "Synthesizer: gemini, excluded from the panel" is right but its panel-shrink consequence is unstated; "no CLI/subscription backend exists" is still true.

---

## 1. Executive synthesis

**Current state.** The deliberation engine is solid and closely matches its documentation: 5 API-SDK providers behind an async `AIProvider` ABC (`providers/base.py:80-107`, `cli.py:38-45`), 4-debater + gemini-synthesizer default runs, blind anonymized critique (`debate.py:19-34`), narrative synthesis with quality-weighting, minority-report artifact on non-unanimous verdicts, per-provider cost metrics, research mode with a degradation alarm (exit 3, ADR-08). Faza 0 delivered the *plumbing* of delegation: `protocols/` as the invocation surface (ADR-09), `--return-dir` deterministic return (ADR-10), inbox slug hygiene (#14), minority report (#15).

**The gap.** What exists is a **human-facing authoring contract** (COUNCIL_QUESTION_GUIDE tells a person how to write a good question) plus **undocumented machine affordances** (`--file`, `--return-dir`, `--format json`, exit codes 0/1/2/3, deterministic artifact names, cwd-independent operation — `config_loader.py:199-201` resolves output relative to repo root *by design*). What does not exist is the **machine-facing invocation contract** that a delegating CC session in another repo can target. The contract is also internally inconsistent across lanes: `--file` mode does not parse frontmatter at all (`cli.py:565-567` — raw `read_text`; frontmatter would leak into the question text), and research mode ignores `--return-dir` entirely (`run_research()` has no such parameter, `research/runner.py:133-143`). `protocols/` is *necessary but not yet sufficient* as the window on the world.

**Single highest-leverage decision: ADR-11 — ratify and document the two-lane delegated invocation contract.** It converts already-shipped Faza-0 plumbing into an actually-usable delegation window, costs one protocol doc plus two small parity fixes, is fully baseline-independent (the same separability argument ADR-10 used for #13), and gives the deferred `/council-question` work (#9) a stable target contract instead of an implicit one.

**Proposed sequence (what unblocks what):**

| Wave | Work | Gate |
|---|---|---|
| 0 | Doc reconciliation: fix GUIDE §synthesizer mechanism text + examples; VISION ADR refs; record the "effective default = 4 debaters + gemini synth" fact | none — hours |
| 1 | ADR-11: `protocols/COUNCIL_INVOCATION_CONTRACT.md` + lane-parity fixes (frontmatter in `--file`, `return_dir` in research) + optionally the `~/.claude council.return_dir` reader (ADR-10 reserved seam) | none — baseline-independent |
| 2 | ADR-12: CliProvider engine (#16) + `backend:` config axis + routing profiles; run the CLI-vs-API parity experiment (scored with SYNTHESIS_QUALITY_RUBRIC) | engine: none; **default flip: parity evidence** |
| — | **Epic B #1→#2 (synthesizer baseline)** — with the D5 scoring guard below | operator-run |
| 3 | #9 `/council-question` template + gate (targets the ADR-11 contract); #18 crux-check (ADR-13 shape); #19 framing alarm; ADR-01/02 amendments (#2/#4); dissent-detection hardening | **Epic B settled** |

---

## 2. Per-area audit

### Area 1 — Invocation surface & delegation flow (operator priority)

**Current-state (verified live).**
- Entrypoints: console scripts `ai-council` and `council` → `ai_council.cli:main` (`pyproject.toml:30-32`); `src/ai_council/__init__.py` is empty — **no programmatic API**, no server/MCP (grep verified). VISION.md:26 declares this deliberate ("Standalone tool — invoked by other repos, not embedded as a library"); VISION.md:38 names "Claude Code in any project under `Dev/`" as a caller.
- The window today = `protocols/COUNCIL_QUESTION_GUIDE.md` (question authoring: frontmatter keys `models/synthesizer/rounds/mode/target-project`, body Question/Current State/Questions/Constraints, research variant) + `SYNTHESIS_QUALITY_RUBRIC.md` (5-criterion scoring) per ADR-09 (Proposed, 2026-07-02).
- Machine affordances that already exist: `--file` (`cli.py:287`), `--return-dir` (`:311`, ADR-10 — copies verdict AND minority report, `orchestrator.py:158-190`), `--format json` (dumps `DebateResult` to stdout, `orchestrator.py:192-197`), exit codes 0/1/2/3 (`cli.py:612-614`, ADR-08), deterministic artifact names `council-out-<ts>-<mode>-<slug>.md` + `_metrics.json` sidecar + `council-minority-*` (`output.py:198,314,403-447`), **cwd-independence** (`config_loader.py:199-201`: output resolved against repo root "regardless of which directory the user runs `council` from"; global secrets loaded from `~/Documents/.secrets/.env`, `cli.py:386-389`).
- Inbox lane: drop `.md` in `council_inbox/` or `~/Downloads` (frontmatter-key sniffing, `settings.yaml:27-32`) → `council --inbox` → parse frontmatter (`cli.py:466-482`, precedence CLI flag > frontmatter > config default) → archive; batch-level exit 3 (`cli.py:464-465,560-561`).
- ADR-67 step 2/3/6 status: `/council-question` template + gate **unbuilt by explicit deferral** (BACKLOG #9: "DEFERRED — do NOT build before the canonical-baseline settles"); `~/.claude council.return_dir` reader **reserved, deliberately unimplemented** (`cli.py:401-406`, ADR-10:44-46). Hub ADR-95 (2026-07-03) confirms the wiring is record-only, ai-council-owned, sequenced-after.

**Gap vs brief/vision.**
- No machine-facing contract document: a delegating agent must reverse-engineer flags, exit codes, artifact names, and lane semantics from code. The guide even anti-recommends CC-prompt elements (`:199-210`) without saying what a CC *caller* should do instead.
- **Lane asymmetry (verified):** direct lane ignores frontmatter (`cli.py:565-567`); inbox lane requires it. A brief written per the guide behaves differently depending on which lane consumes it — in `--file` mode its frontmatter pollutes the question text sent to panelists.
- **Research-mode return gap (verified):** `--return-dir` silently no-ops for research (`research/runner.py:133-143` — no param; grep `return_dir` in `research/` = zero hits). A repo commissioning research cannot get a deterministic return.
- Brief §6.1 (interaction model fork): unresolved — both lanes exist, neither is blessed as *the* delegation path.

**Decision.** *(actionable-now)* Bless a **two-lane model** and write it down as a machine contract:
- **Lane A — delegated/synchronous:** external CC session runs `council --file <brief.md> --return-dir <dir> [--format json] [flags]` from its own repo; consumes exit code + JSON + artifacts at the return dir. This is the "window on the world" for agents.
- **Lane B — inbox/batch:** operator-mediated fire-and-forget, unchanged (preserves brief §6.1's batch model; no interactive/context-pull concepts — those stay cut per brief §5).
- Close the two parity gaps so one brief format works in both lanes: parse frontmatter in `--file` mode via the same `parse_file()` used by the inbox (flags > frontmatter > defaults, same precedence as `cli.py:473-482`), and thread `return_dir` through `run_research()`.
- Trade-off accepted: blessing Lane A makes the CLI surface a public ABI — future flag changes become breaking changes governed by the contract doc. That is the point.

**ADR-draft:** ADR-11 (§3 below).

---

### Area 2 — Backend / cost model

**Current-state (verified live).**
- All 5 debate providers + 5 research providers are API-SDK based: `PROVIDER_CLASSES` (`cli.py:38-45`) → `AsyncAnthropic` / `AsyncOpenAI` (+ base_url variants for xai/deepseek) / `google.genai`. **Zero subprocess usage in `src/`** (grep verified; only git-gate scripts use subprocess). The interface is `async def generate(prompt, round_number) -> ModelResponse` (`providers/base.py:93`).
- Cost model: per-provider `cost_per_1m_input/output` in `settings.yaml:34-91`; per-call metrics (`metrics.py`); ~$0.50/debate and "cost gate lives in operator judgment, not policy" (hub AI_COUNCIL_PROCESS.md:394); a 6th provider `claude-sonnet` exists in `all_providers` but sits on no default panel — an unused cheap seat.
- BACKLOG #16 (Epic C, P3/L) is the CliProvider story [Rama 2], with design tensions listed (read-only sandbox, non-determinism, anonymization, quota-vs-devwork) "deferred to build-start" and the explicit constraint **do NOT merge provider implementations**.
- **Honesty note:** the CLI-flag recon the prompt attributes to Epic C is absent from the repo (correction #4 above). The operator-supplied flags are plausible but must be re-verified at build time — CLI surfaces of claude/codex/gemini change fast.

**Gap vs brief/vision.**
- Operator intent (CLI-subscription cheap lane for most debates; API reserved for important debates + research) has **no representation anywhere**: no backend axis in config, no routing policy, no quota awareness. Brief §6.2's fork (API-$ no-ceiling vs subscription quota shared with dev work) is undecided.

**Decision.** *(engine: actionable-now · default flip: evidence-gated)*
- **Engine:** add a `backend: api|cli` axis per provider block in `settings.yaml`; implement `CliProvider` adapters **behind the existing `AIProvider` ABC** (one adapter per CLI, `asyncio.create_subprocess_exec`, structured-JSON stdout, hard timeout reusing `timeout_sec`). Non-negotiables: read-only sandbox flags on every invocation (operator safety invariant); personas/mode directives injected via the prompt (defeats the CLI's own coding-assistant system prompt as far as possible — residual contamination is a known, accepted risk logged per-run); responses enter the normal pipeline so ADR-03 anonymization (our-side shuffle + relabel) applies unchanged — residual style leakage already exists between API models and is not newly introduced. Each adapter is a separate class; xai/deepseek stay API-only (no CLI exists) — respects the no-merge rule.
- **Routing policy:** profile-based, not per-call heuristics. `standard` profile → CLI backend where available (claude/gemini/codex→openai seats), API elsewhere; `important` profile (flag or frontmatter key) → all-API; **research mode → always API** (research providers are API-only products). Fallback: on CLI failure/quota error (extend `classify_error()`, which already has a `billing` category, `providers/base.py:26-34`), retry the same seat via API. This makes the orchestrator the brief's "three-axis allocator" in the simplest viable form: a static profile + failure fallback, no dynamic quota accounting in v1 (quota starvation of dev work is mitigated by profile choice remaining in operator hands, honoring "cost gate lives in operator judgment").
- **The default flip** (standard debates default to CLI lane) ships only after a parity run: N paired debates (same briefs, CLI-backed vs API-backed panels) scored with `protocols/SYNTHESIS_QUALITY_RUBRIC.md`. This is **its own evidence gate, not the Epic B baseline gate** — it reuses the rubric, not the synthesizer decision.
- Trade-off: CLI seats add non-determinism and a new failure surface to a deliberately deterministic tool; contained by keeping API as the fallback and `important`/research API-only.

**ADR-draft:** ADR-12 (§3 below).

---

### Area 3 — Council epistemic mechanics

**Current-state (verified live).**
- **#15 minority report — SHIPPED, do not re-propose.** `extract_dissent()` + `save_minority_report()` (`output.py:381-447`), invoked at `orchestrator.py:177-190`, routed to all destinations including `--return-dir`. Detection is **heuristic**: scans synthesis for headings matching `("unresolved disagreement","contested point","dissent","minority")` with a genuine-body filter (`output.py:321-378`). There is **no structured vote tally** — "blind voting" (ADR-03) is in fact blind *critique*; the verdict is the synthesizer's narrative (confirmed in code comment `output.py:381-389`).
- **#13 output routing — SHIPPED** (ADR-10, `output.py:121-176` `_write_routed`: canonical always-first; hub never a default). **#14 double-council fix — SHIPPED** (`inbox.py:28-50`).
- **Synthesizer identity — see correction #1/#2.** Mechanism verified: default runs = 4 debaters + gemini non-participant synthesizer. Doc drift verified: GUIDE `:481` mis-describes the mechanism; GUIDE examples hardcode `synthesizer: openai`; ADR-02/ARCHITECTURE say "full 5-model" without stating the shrink. Consequence for Epic B: **historical transcripts may be openai-authored** (inbox runs honoring guide-example frontmatter, `cli.py:478-482`), which contaminates #1's premise of scoring "the current Gemini synthesizer against ~15 historical transcripts."
- **Regime split (brief L3):** the June-16 audit already encodes it rigorously (mode-split table, per-regime gap scoping). The rubric measures the subjective-regime metric (dissent surfacing, faithfulness) — correctly aligned.
- **#18 (crux resolution, Rama 1) and #19 (framing defense, Rama 3):** already backlogged, both tagged baseline-gated (Epic B). G6 remains real: `pick`/`judge` debaters have zero retrieval (`research/runner.py` pool is research-mode-only).

**Gap vs brief/vision.**
- Docs contradict the verified mechanism (three places: GUIDE rule text, GUIDE examples, ADR-02/ARCHITECTURE panel-size framing). ADR-01's declared default is real *at runtime* but not *in practice* whenever briefs follow the guide's examples — the verdict author varies by invocation habit, invisibly.
- Minority-report detection is coupled to the synthesizer's heading discipline; a synthesizer swap (Epic B Branch A) could silently change dissent-emission behavior.
- Tool-grounded crux resolution (the brief's keystone L4) — absent, correctly gated.

**Decision.** *(doc fixes: actionable-now · rest: baseline-gated)*
- Fix the three doc sites now (mechanism description, examples aligned to the actual default, an explicit "effective default: 4 debaters + gemini synthesizer" statement). Formal ADR-01/ADR-02 amendments stay with Epic B #2/#4 (which already own "amend ADR-01" and "panelist/synthesizer overlap policy") — do not pre-empt them.
- **Epic B #1 scoring guard (new, cheap, actionable-now):** before scoring, read each historical transcript's synthesizer identity from its header/metrics and segment results by verdict author. Without this, #1's branch decision is built on contaminated data.
- **Crux-check shape** (for #18, recorded now so build doesn't re-litigate): a *bounded, discrete step between rounds* with a pluggable resolver — v1 grounds flagged empirical cruxes via the existing research pool (bounded single-shot, cached via `make_cache_key`), v2 swaps in a CLI-agent resolver when ADR-12 lands (the brief's "agents solve Rama 1 for free"). Fail-open: resolution failure never blocks the debate; evidence is injected anonymously into the round-2 prompt. Stays within rounds≤2 (it is not a round).
- Minority-detection hardening (ask the synthesizer for an explicit machine-readable dissent block) folds into Epic B's synthesis-contract work — it touches exactly the prompt Epic B is about to re-evaluate.

**ADR-draft:** ADR-13 (shape only, explicitly not-to-ratify before Epic B). Doc fixes need no ADR.

---

### Area 4 — Process ownership

**Current-state (verified live).**
- Hub owns the process: `AI_COUNCIL_PROCESS.md` v2.0 + ADR-67, whose "Where each piece lives" table assigns **template + gate + known-path I/O to ai-council** and the `council.return_dir` key to `~/.claude`, grounded in the ADR-28 layer invariant ("this repo holds process specs, not executable gate logic"). ADR-95 (2026-07-03) adds the lane split: architect frames substance; CC mechanically expands — and confirms `/council-question` wiring is ai-council-owned, record-only so far.
- Repo owns the tool surface: `protocols/` (ADR-09), mirroring the hub's own protocols pattern.
- **Hub-side staleness (verified):** `AI_COUNCIL_PROCESS.md:11-17` still points at `ai-council/docs/council-question-guide.md` — the file moved to `protocols/COUNCIL_QUESTION_GUIDE.md` under ADR-09. ADR-43 establishes the amendment pattern for cross-repo semantics: design-stage flag-back to the hub before merging.

**Gap vs brief/vision.**
- The boundary is already right in principle; what's missing on the repo side is the machine contract (Area 1). No hub ADR needs superseding.

**Decision.** *(actionable-now — flags only; no hub edits from this repo)*
- Codify the boundary in the new contract doc: **hub owns WHEN/WHY** (convene thresholds, six-step loop, caps, cost judgment); **ai-council `protocols/` owns HOW** (question format, invocation mechanics, output contract, quality rubric).
- File two hub-reconciliation flags for a hub-side session (never edit ADR-60/67 or hub files from here): (1) the stale `docs/council-question-guide.md` pointer in AI_COUNCIL_PROCESS.md §authoritative-sources; (2) once ADR-11 ratifies, Stage 3's "Command:" section should name Lane A alongside `--inbox` — flag at design stage, extending the ADR-43 feedback-edge courtesy to invocation semantics.

**ADR-draft:** none — folded into ADR-11's reconciliation section.

---

### Area 5 — End-to-end pipeline

**Current-state (verified live) — the whole flow as it runs today:**

```
external repo (CC)                    ai-council                              return
──────────────────                    ──────────────────────────────────      ─────────────────
architect frames (ADR-95)
→ author brief per GUIDE     →  Lane B: council_inbox/ | ~/Downloads
   (frontmatter: models/           → operator: council --inbox
   synthesizer/rounds/mode/        → parse_file → RunRequest
   target-project)              Lane A: council --file brief.md [flags]
                                   (frontmatter NOT parsed — gap)
                                → health gate → panel (synth evicted:
                                   4 debaters default) → rounds ≤2
                                   (blind critique) → synthesis (gemini)
                                → output/: council-out-*.md + _metrics.json
                                   [+ council-minority-* if dissent]
                                → copies: --return-dir | --target-project   →  caller reads
                                → --format json → stdout; exit 0/1/2/3          known path
                                research mode: separate path, API-only,
                                   exit 3 alarm; NO return-dir (gap)
→ CC drafts ADR in target repo (ADR-67 step 5) → close-out
```

**Gap vs brief/vision — the true gaps for real delegation, ranked:**
1. No machine contract for Lane A (Area 1) — the only gap that blocks delegation *today*.
2. Lane-parity defects: `--file` frontmatter; research `return_dir` (Area 1).
3. `/council-question` template + gate — deferred by design (#9); its absence means Stage 1a stays manual, which the hub explicitly tolerates ("perform this review manually").
4. Cost lanes (Area 2) — delegation works without them; they change the economics, not the capability.
5. Epistemic upgrades (#18/#19) — gated; they change verdict *quality*, not delegation.
6. Doc drift (Area 3) — cheap, corrosive if left.

**Decision.** *(actionable-now)* Reaffirm **CLI-as-ABI**: no Python library API, no server/MCP surface (VISION:26 conformant; statelessness preserved; `--format json` + exit codes + known paths are the machine interface). Sequence the work as Waves 0–3 (§1). The pipeline needs no structural change — it needs its contract written down and two parity holes closed.

**ADR-draft:** none beyond ADR-11/12/13 (the sequence is operator scheduling, not architecture).

---

## 3. ADR drafts (review artifacts — NOT written to docs/decisions/, NOT committed)

Numbering continues from the live set (ADR-01..10 verified present; 09/10 newest). Hub ADRs 60/67 (and 95, 43) treated as immutable; each draft carries an explicit reconciliation section. All three are **drafts for operator + architect review**.

---

### DRAFT `ADR-11-delegated-invocation-contract.md`

```markdown
# ADR-11: Delegated Invocation Contract — two lanes, one machine-readable surface

- **Date:** 2026-07-04 (draft — Fable audit)
- **Status:** Proposed (draft — not ratified, not committed)
- **Related:** ADR-09 (protocols/ surface), ADR-10 (return-dir), ADR-08 (exit codes);
  hub ADR-67 (gated loop — immutable), hub ADR-95 (lane split — immutable), hub ADR-43 (routing — immutable)

## Context
ai-council is commissioned by external repos (VISION: "Called by other repos … via the
`council` CLI"). Faza 0 shipped the plumbing (protocols/, --return-dir, minority report),
but the machine-facing contract is implicit: a delegating agent must reverse-engineer
flags, exit codes, and artifact names from code. Two verified lane-parity defects make the
implicit contract inconsistent: (1) `--file` mode does not parse frontmatter (cli.py:565-567)
— a guide-conformant brief behaves differently per lane, and its frontmatter leaks into the
question text; (2) research mode ignores `--return-dir` (run_research has no such parameter).

## Decision
1. **Two invocation lanes, both first-class:**
   - **Lane A (delegated, synchronous):** `council --file <brief.md> --return-dir <dir>
     [--format json] [overrides]`, runnable from any cwd (config already resolves paths
     against repo root by design). The lane for agent callers.
   - **Lane B (inbox, batch):** unchanged operator-mediated fire-and-forget
     (`council --inbox`). The interaction model stays batch — no context-pull /
     interactive concepts (consolidation-brief §5 cuts upheld).
2. **One brief format across lanes:** `--file` parses YAML frontmatter through the same
   `parse_file()` path as the inbox, with identical precedence: CLI flag > frontmatter >
   config default. Frontmatter is stripped from the question text.
3. **Research parity:** `run_research()` gains `return_dir`, routed through the research
   output writer, same semantics as debate outputs (canonical ./output/ always; return-dir
   is an additional copy; hub never a default).
4. **The contract is documented as `protocols/COUNCIL_INVOCATION_CONTRACT.md`** covering:
   both lanes; full flag set; frontmatter keys + precedence; exit codes 0/1/2/3 (ADR-08);
   artifact naming (`council-out-*`, `council-minority-*`, `*_metrics.json`); `--format
   json` stdout payload (DebateResult schema); degradation semantics; RoutingError
   fail-loud; ownership boundary (hub owns WHEN/WHY — convene rules, loop, caps; this
   repo's protocols/ owns HOW — formats, mechanics, outputs).
5. **CLI-as-ABI reaffirmed:** no Python library API, no server/MCP. The CLI surface named
   in the contract doc becomes a compatibility surface — breaking changes require an ADR.

## Considered and rejected
- Python library API / MCP server — contradicts VISION ("not embedded as a library"),
  adds a stateful surface to a deliberately stateless tool, and duplicates what exit
  codes + JSON + known paths already provide.
- Single-lane (inbox-only) delegation — leaves the delegating agent unable to run
  synchronously and consume results programmatically; contradicts ADR-08's own rationale
  (automation harness reading exit codes).

## Reconciliation with hub ADRs (no edits, no supersession)
- **ADR-67:** implements its step-4/6 "known-path I/O" assignment to ai-council; the
  ~/.claude `council.return_dir` reader remains ADR-10's reserved seam (legal future
  setter; implementing it needs no new ADR). No divergence.
- **ADR-95:** the contract is exactly the surface "CC mechanically expands" against. No divergence.
- **ADR-43:** target-project routing semantics untouched. Courtesy design-stage flag-back
  to the hub anyway (see flags), extending ADR-43's feedback edge to invocation semantics.
- **Hub flags (hub-side session, not this repo):** stale guide path in
  AI_COUNCIL_PROCESS.md §authoritative-sources; Stage-3 "Command:" wording once this ADR
  ratifies.

## Consequences
- External repos can commission the Council today, without waiting for Epic B or #9;
  #9's template/gate, when built, targets this contract instead of an implicit one.
- Baseline-INDEPENDENT (same separability argument as ADR-10/#13).
- Cost: one protocol doc + two small parity fixes + tests; the CLI surface becomes a
  versioned commitment.
```

---

### DRAFT `ADR-12-provider-backend-engine-and-cost-lanes.md`

```markdown
# ADR-12: Provider Backend Engine — CLI-subscription seats and two-lane cost policy

- **Date:** 2026-07-04 (draft — Fable audit)
- **Status:** Proposed (draft — engine actionable; default-flip clause evidence-gated)
- **Related:** BACKLOG #16 [Rama 2]; ADR-02 (panel), ADR-03 (anonymization), ADR-06 (cost);
  hub AI_COUNCIL_PROCESS (cost gate = operator judgment — unchanged)

## Context
All providers are API-SDK based (PROVIDER_CLASSES, cli.py:38-45; zero subprocess in src/ —
verified). The operator pays for Claude Code / Codex / Gemini subscriptions the council
never uses. Intent: CLI-subscription lane for most debates; API reserved for important
debates + research. NOTE: no CLI-flag recon exists in-repo; operator-supplied flags
(claude -p --output-format json --tools ""; codex exec --sandbox read-only --json;
gemini -p -o json) are input to this design and MUST be re-verified at build start.

## Decision
1. **Backend axis:** each `models:` block in settings.yaml gains `backend: api | cli`
   (default `api`). Config-only — no hardcoding (invariant 3 upheld).
2. **CliProvider adapters** behind the existing `AIProvider` ABC — one adapter per CLI
   (claude / codex→openai seat / gemini), `asyncio.create_subprocess_exec`, structured
   JSON stdout, `timeout_sec` reused as hard kill. Separate classes; xai/deepseek remain
   API-only. The no-merge rule (CLAUDE.md §5.7, #16) is upheld.
3. **Safety invariants (non-negotiable):** every CLI invocation runs read-only /
   tools-disabled (per-CLI sandbox flags verified at build); cwd = a scratch dir, never a
   repo; personas + mode directives injected via the prompt; provider output enters the
   normal pipeline so ADR-03 anonymization (shuffle + relabel, our side) applies
   unchanged. Residual risks accepted and logged per-run: CLI system-prompt
   contamination, style leakage (pre-existing across API models), non-determinism.
4. **Routing policy — profiles, not heuristics:**
   - `standard` (default *after* the flip clause): CLI backend where available, API elsewhere.
   - `important` (CLI flag / frontmatter key): all-API.
   - research mode: always API (research providers are API-only products).
   - Fallback: CLI failure or quota error (extend classify_error's categories) retries
     the same seat via API — quota exhaustion degrades cost, never the debate.
   The cost gate stays in operator judgment (hub doctrine unchanged); v1 does no dynamic
   quota accounting — profile choice is the quota control.
5. **Default-flip clause (evidence-gated):** `standard` defaults to the CLI lane only
   after a parity run — N paired debates (same briefs, CLI-backed vs API-backed), scored
   with protocols/SYNTHESIS_QUALITY_RUBRIC.md, no material quality regression. Until
   then `backend: api` remains the default everywhere. This is the parity-evidence gate,
   NOT the Epic B baseline gate — it reuses the rubric, not the synthesizer decision.

## Considered and rejected
- Judge-centric cost shape (cheap CLI panel + one expensive API judge) as the *default* —
  concentrates epistemic authority (brief #6's own tension); remains expressible via
  config once the axis exists, so it needs no decision now.
- Dynamic per-call quota allocator — three-axis accounting complexity without evidence it
  is needed; profiles + fallback cover the failure mode.

## Reconciliation
- ADR-02 panel composition untouched (seats change transport, not membership).
- ADR-03 unaffected — anonymization is applied by the debate engine, not the provider.
- ADR-06 extended, not superseded: this is the cost-optimization lane it anticipated.
- Brief §6.2 fork resolved as: subscription-first is a *profile*, not a posture change;
  API remains the no-ceiling fallback.

## Consequences
- Rama-1 bonus unlocked: a CLI agent seat is a candidate crux RESOLVER (ADR-13 v2).
- New failure surface (subprocess, JSON parsing, CLI version drift) — contained by the
  API fallback and by `important`/research staying API-only.
- BACKLOG #16 done-when is satisfied by the first CLI backend running a debate turn
  through the provider protocol.
```

---

### DRAFT `ADR-13-bounded-crux-check.md` — **baseline-gated; do not ratify before Epic B**

```markdown
# ADR-13: Bounded Crux-Check Step — tool-grounded resolution of empirical cruxes

- **Date:** 2026-07-04 (draft — Fable audit)
- **Status:** Proposed (draft — BASELINE-GATED on Epic B; recorded now so #18's build
  does not re-litigate the shape)
- **Related:** BACKLOG #18 [Rama 1]; June-16 audit G6; ADR-12 (CLI resolver option);
  hub ADR-67 rounds≤2 cap (immutable — respected)

## Context
Decision-mode debates have zero retrieval (research pool is research-mode-only —
verified); empirical sub-claims inside pick/judge debates are settled by the most
confident voice (audit G6; persuasion ≠ truth). This is the consolidation brief's
keystone (L4): the one gap research, audit, and practice converge on.

## Decision (shape only)
1. **A discrete, bounded crux-check step between Round 1 and Round 2** — not a new round
   (rounds≤2 cap untouched), not open-ended research inside a debate.
2. **Flagging:** the Round-1/critique prompt gains a "checkable claims" field; the
   orchestrator collects flags, caps at 3 cruxes per debate, pick/judge modes only.
3. **Pluggable resolver** behind a small `CruxResolver` protocol:
   - v1: single-shot bounded lookup via the existing research pool, cached via
     make_cache_key (reuse, no new machinery);
   - v2: a read-only CLI-agent check once ADR-12 lands (an agent can RUN a check; an
     endpoint can only assert one).
4. **Injection:** resolved evidence enters the Round-2 prompt as an anonymous evidence
   block — attributed to no panelist (ADR-03 spirit), marked as tool-derived.
5. **Fail-open:** resolver timeout/failure never blocks the debate; the transcript
   records "crux flagged, unresolved" — which is itself signal for the verdict.

## Why gated
Epic B re-decides the verdict author and synthesis contract; the crux-check changes the
epistemic inputs to exactly that contract. Building it against a baseline about to move
would double the evaluation confound. Gate: Epic B #1/#2 settled.

## Reconciliation
- ADR-67 caps upheld (step ≠ round; debates ≤2 unchanged).
- Brief tensions honored: bounded (cost ceiling per debate ≈ one research lookup),
  deterministic-ish (cached), batch model untouched.
```

---

## 4. Tagged decision list (for operator sequencing)

Tags: **[NOW]** actionable-now · **[EVID]** gated on its own experiment (not Epic B) · **[BASE]** baseline-gated (Epic B synthesizer/epistemic baseline).

| # | Decision | Tag | Vehicle |
|---|---|---|---|
| D1 | Bless two-lane invocation contract; author `protocols/COUNCIL_INVOCATION_CONTRACT.md` | **[NOW]** | ADR-11 |
| D2 | Lane-parity fixes: frontmatter parsing in `--file`; `return_dir` through research mode | **[NOW]** | ADR-11 §2-3 |
| D3 | `~/.claude council.return_dir` reader (ADR-10 reserved seam; low priority) | **[NOW]** | ADR-10 seam — no new ADR |
| D4 | Synthesizer-identity doc reconciliation (GUIDE `:481` mechanism + examples; VISION ADR list; state "effective default = 4 debaters + gemini synth") | **[NOW]** | doc fix — no ADR |
| D5 | Epic B #1 scoring guard: verify per-transcript verdict author before scoring; segment by synthesizer | **[NOW]** | note into #1's done-when |
| D6 | CliProvider engine + `backend:` config axis + routing profiles | **[NOW]** | ADR-12 / #16 |
| D7 | Flip `standard`-debate default to CLI lane | **[EVID]** (rubric parity run) | ADR-12 §5 |
| D8 | Reaffirm CLI-as-ABI: no library API, no server/MCP | **[NOW]** (decision only) | ADR-11 §5 |
| D9 | `/council-question` template + question-quality gate | **[BASE]** (deferral upheld) | #9 → targets ADR-11 contract |
| D10 | Crux-check step shape (bounded, pluggable resolver, fail-open) | **[BASE]** | ADR-13 / #18 |
| D11 | Active framing defense + false-consensus alarm | **[BASE]** | #19 — no ADR yet |
| D12 | ADR-01/ADR-02 amendments (verdict author; panel-size/overlap policy) | **[BASE]** | Epic B #2/#4 — not pre-empted |
| D13 | Minority-detection hardening (structured dissent block in synthesis contract) | **[BASE]** | fold into Epic B synthesis work |
| D14 | Hub reconciliation flags (stale guide path; Stage-3 wording post-ADR-11) — hub-side session only | **[NOW]** (flag only) | ADR-43-style design-stage flag-back |

**Dependency spine:** D1/D2 unblock external delegation and give D9 its target → D6 unblocks D7 and ADR-13's v2 resolver → Epic B (with D5's guard) unblocks D9–D13.

---

## 5. Boundaries respected / not proposed

- **Nothing re-proposed from Faza 0:** protocols/ surface (ADR-09), return-dir routing (#13/ADR-10), double-council fix (#14), minority report (#15) — verified shipped, cited above.
- **Brief §5 cuts upheld:** no iterate-to-convergence, no automated debate-gating, no cross-session reputation, no context-pull/thinking-prosthesis. Brief §6.1 resolved as "stay batch"; §6.2 resolved as profiles; §6.3 resolved as "pure governance" (no scope extension).
- **No hub edits proposed** — ADR-60/67/95/43 treated as immutable; divergences = none; two hub-side flags filed for a hub session (D14).
- **#9's explicit deferral respected** — nothing in Waves 0–2 builds the template/gate.
- **Nothing implemented, no files changed, no commits** — this document is the only artifact.

## How to verify this audit (spot-checks for the reviewer)

1. Synthesizer trace: `orchestrator.py:61-74` → `runner.py:44-56` then `:59-75`; run mentally with panel=`full_panel`, preferred=`gemini`.
2. Lane asymmetry: `cli.py:565-567` (raw read; no `parse_file`) vs `cli.py:466-482` (inbox parse + precedence).
3. Research return gap: `grep return_dir src/ai_council/research/` → zero hits; `run_research` signature `research/runner.py:133-143`.
4. No CLI backends: `grep -rE "subprocess|Popen" src/` → no provider-layer hits.
5. Guide drift: `protocols/COUNCIL_QUESTION_GUIDE.md:481` vs the code trace in (1); examples at `:46-48`, `:511-515`.
6. Epic C recon absence: grep worktree for `codex exec` / `--output-format` → BACKLOG #16 story text only.
