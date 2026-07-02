# BACKLOG — ai-council

## Big picture

`ai-council` is the ecosystem's multi-model deliberation engine: it runs structured
debate/research across a configurable AI panel and produces the verdicts that become
binding ADRs. The backlog advances the tool toward a delegation-ready, evidence-based,
reliable, and self-enforcing state across six epics.

**Epics (backbone):** A Invocation surface & delegation-readiness · B Synthesizer refresh ·
C Provider reliability & CLI engine · D Model currency · E Naming & quality automation ·
F Council process & epistemic quality

---

## Epic A — Invocation surface & delegation-readiness
> As the tool owner, I want ai-council's invocation specs and outputs to live in a clean,
> delegation-ready surface, so an external agent/operator can commission the Council without
> ambiguity about where specs live or where results land.

### Give the invocation specs and outputs a first-class home
So that the tool is commissionable and its outputs are predictable and durable.
- [#12] [P2][M] Create top-level `protocols/` invocation surface: `git mv docs/council-question-guide.md → protocols/COUNCIL_QUESTION_GUIDE.md` and `docs/synthesis-quality-rubric.md → protocols/SYNTHESIS_QUALITY_RUBRIC.md` (SCREAMING_SNAKE, mirroring the hub); `docs/` becomes decisions + audits + archive only. Update the `.pre-commit-config.yaml` hub-hook path (`toc-freshness`/`toc-generate`: `^docs/council-question-guide\.md$` → `^protocols/COUNCIL_QUESTION_GUIDE\.md$`), ARCHITECTURE Folder Governance, CLAUDE.md §10 ref, BACKLOG #10 ref, and every living cross-ref (grep both slugs first) · Done when: files live under `protocols/`, hook + all living refs updated, `check.ps1` green · refs ADR-09
- [#13] [P2][M] Output routing: default output to local `./output/` when `return_dir` is unset; `--return-dir <path>` CLI override; **never** the hub as default; update all docs · Done when: unset → `./output/`, `--return-dir` routes to the given path, hub never a silent default · refs ADR-10 · (= the `return_dir` I/O piece split out of #9; baseline-INDEPENDENT)
- [#14] [P3][S] Fix the double-"council" output filename: `clean_slug()` strips `FAILED_`/timestamp prefixes but not a leading "council", yielding `council-out-…-council-…` · Done when: emitted filenames carry a single "council" token · refs `clean_slug()` in `src/ai_council/inbox.py`; adjacent to #8 (same emitter)
- [#15] [P3][M] Minority report as a first-class output [Rama 4]: emit dissent as a discrete, durable artifact alongside the verdict (same output subsystem as #13) · Done when: a run with genuine dissent produces a separate durable minority-report artifact · refs Rama 4

---

## Epic B — Synthesizer refresh
> As the tool owner, I want the default synthesizer chosen on real scoring data and the choice codified, so the verdict author is evidence-based and cost-aware.
> **Note:** this epic is the baseline gate for Epic F's baseline-gated items.

### Decide the synthesizer on real data, then codify it
So that the ADR-01 default rests on measured synthesis quality, not assumption.
- [#1] [P1][M] Run the current Gemini synthesizer against ~15 historical transcripts and score with the synthesis rubric · Done when: ~15 transcripts scored + the Phase-3 branch trigger is decided (Branch A swap / Branch B keep) · refs Phase-2 smoke test
- [#2] [P1][M] Implement the Phase-3 conditional: amend ADR-01 (cost-optimization principle) + execute Branch A (new synthesizer) or Branch B (keep Gemini) · Done when: ADR-01 amended + the chosen branch shipped · refs ADR-01 · BLOCKED on #1 (needs the smoke-test scores)
- [#3] [P1][S] Codify the cost-optimization principle in the ADR-01 amendment text (balance quality vs cost) · Done when: the principle is written into the ADR-01 amendment · refs ADR-01 (folded into #2 scope, tracked separately)

### Settle the panelist/synthesizer overlap policy
So that overlap rules are explicit if the synthesizer ever joins the panel.
- [#4] [P3][S] Amend ADR-02 to codify the cost-reframed panelist/synthesizer overlap policy · Done when: ADR-02 amendment lands (or is closed as not-needed if Gemini retained) · refs ADR-02 · conditional on #2 Branch A

---

## Epic C — Provider reliability & CLI engine
> As the tool owner, I want every wired provider to have a known-good, tested path and a route to CLI-subscription backends, so reliability is measured and cost is controllable.

### Close the untested and unreliable provider paths
So that no provider is wired-but-unverified or silently degrading the panel.
- [#5] [P2][M] Add an integration path for `openai_deep_research.py` (o3-deep-research, ~45 min, ~$10+/run — cannot run in CI) · Done when: a manual integration test path is documented + runnable · refs migrated from tasks/todo.md
- [#6] [P3][M] Evaluate whether DeepSeek should be replaced or demoted from the default full panel · Done when: a replace/keep/demote decision is recorded · refs reactive trigger: round-blocking failure rate >2% per JOURNAL data
- [#20] [P3][S] Fix OpenAI-SDK Responses-API type-stub drift: 6 mypy errors in `src/ai_council/research/providers/{openai_mini,openai_deep,grok}_research.py` (`.create()` overloads, object-not-iterable) · Done when: mypy clean on those files + `check.ps1` fully green · refs pre-existing on main, surfaced 2026-07-02 during Unit 1 merge

### Add a CLI-subscription provider backend
So that subscription CLIs can serve debate turns and API spend is reserved for CLI-less models.
- [#16] [P3][L] CliProvider engine [Rama 2]: an adapter behind the provider protocol that drives CLI backends (Claude/Gemini/Codex subscriptions); API access reserved for CLI-less providers (DeepSeek/Grok) · Done when: at least one CLI backend runs a debate turn through the provider protocol · refs Rama 2 · design tensions (read-only sandbox, non-determinism, response anonymization, quota-vs-devwork contention) deferred to build-start · do NOT merge provider implementations (keep separate)

---

## Epic D — Model currency
> As the tool owner, I want to know when the configured panel models fall behind the latest releases, so the Council never silently debates on stale models.

### Detect stale model configuration
So that a superseded model in settings.yaml is surfaced, not silently used.
- [#17] [P3][M] Online model-version check: verify the `config/settings.yaml` model strings are the latest available per provider + a documented update process · Done when: a check reports any configured model that is no longer the latest + the refresh process is written down · refs `config/settings.yaml` is the single source of model strings

---

## Epic E — Naming & quality automation
> As the tool owner, I want the ADR-34 naming convention enforced mechanically and its edge cases resolved, so violations are caught by CI, not reviewer luck.

### Enforce hyphen-only naming and resolve its timestamp edge case
So that new files cannot drift from ADR-34 and the ISO-timestamp ambiguity is settled.
- [#7] [P2][M] Add a CI check (pre-commit hook or ruff plugin) rejecting new `docs/`/`src/` files with UPPERCASE or underscores in the slug · Done when: a non-conforming new filename is blocked · refs strażnik finding I5 (SYNTHESIS-QUALITY-RUBRIC.md violation)
- [#8] [P3][S] Decide whether ADR-34 applies to ISO timestamps inside filenames (the `council-out-YYYYMMDD_HHMMSS` underscore) — fix the emitter to hyphens, or amend ADR-34 with an ISO-timestamp exemption · Done when: the methodology decision is recorded + applied · refs ADR-34 · surfaced by #7 once built · adjacent to #14 (same emitter)

---

## Epic F — Council process & epistemic quality
> As the tool owner, I want the ADR-67 gated loop implemented, the synthesis rubric sharpened, and the panel's epistemics defended, so the Council runs deterministically and resists framing bias and false consensus.

### Build the ADR-67 downstream pieces and sharpen the rubric
So that `/council-question` generates + self-gates questions and synthesis quality is unambiguous.
- [#9] [P3][L] Implement ai-council's ADR-67 pieces: the `/council-question` template (one decision + options + constraints + prior-ADR context) and the question-quality gate · Done when: `/council-question` generates a templated question and the gate correctly passes/fails it · refs ADR-67 · DEFERRED — do NOT build before the canonical-baseline settles (mirrors `.dev-knowledge` #70) · NOTE: the deterministic `council.return_dir` I/O clause was moved out to #13 (baseline-INDEPENDENT)
- [#10] [P3][S] Refine the faithfulness criterion in `protocols/SYNTHESIS_QUALITY_RUBRIC.md` to clarify additive meta-analysis cases · Done when: the rubric wording disambiguates synthesizer cross-model synthesis vs raw-transcript content · refs N=1 scoring exercise
- [#11] [P3][S] Provide ai-council's two data points (cycle-1 + cycle-2 retrospective) for the "bilateral handshake = 1 round trip" codification owned by `.dev-knowledge` · Done when: the data points are handed to `.dev-knowledge/LESSONS.md` · refs cross-stream (codification lives in `.dev-knowledge`)

### Defend the panel's epistemics
So that deadlocks resolve on evidence and consensus is genuine, not a framing artifact.
- [#18] [P3][M] Tool-grounded crux resolution [Rama 1, baseline-gated]: when the panel deadlocks on a factual crux, resolve it by grounding in a tool/evidence lookup rather than more debate · Done when: a factual crux triggers a grounded lookup that feeds the next round · refs Rama 1 · baseline-gated (Epic B)
- [#19] [P3][L] Active framing defense + false-consensus alarm [Rama 3, runtime-coupled, baseline-gated]: detect and counter leading/asker-leaked framing at runtime and alarm when apparent consensus is an artifact of framing rather than genuine agreement · Done when: a framing-biased question is flagged + a false-consensus run raises an alarm · refs Rama 3 · runtime-coupled; baseline-gated (Epic B)

---

**About this file** — ADR-66 story-map (Big Picture → Epic → User Story → Task), migrated
2026-06-02 from the ADR-41/47 stream schema per ADR-38 A6 (canonical backlog form, all
repos). Stories are human (goal + `So that`); tasks carry `[#id] [P][size] · Done when · refs`.
Done tasks **leave** (ADR-65); git is the implementation record. Conformance is checked
read-only by `.dev-knowledge/scripts/audit.py`.

**Grooming log:** 2026-05-12 (stream-format seed) · 2026-06-02 (story-map migration, all 11 items preserved) · 2026-07-02 (6-epic reorganization: 4 themes → 6 epics A–F; #12–#19 added; #9 re-sliced (return_dir I/O → #13); #20 filed under Epic C — pre-existing mypy drift surfaced during Unit 1; all 11 prior items preserved). Next quarterly: 2026-10-01.
