# BACKLOG — ai-council

## Big picture

`ai-council` is the ecosystem's multi-model deliberation engine: it runs structured
debate/research across a configurable AI panel and produces the verdicts that become
binding ADRs. The backlog advances the tool toward a delegation-ready, evidence-based,
reliable, and self-enforcing state across six themes.

**Themes (backbone) — epic ids:** [E1] Invocation surface & delegation-readiness · [E2] Synthesizer refresh ·
[E3] Provider reliability & CLI engine · [E4] Model currency · [E5] Naming & quality automation ·
[E6] Council process & epistemic quality

---

## [E1] Invocation surface & delegation-readiness
> As the tool owner, I want ai-council's invocation specs and outputs to live in a clean,
> delegation-ready surface, so an external agent/operator can commission the Council without
> ambiguity about where specs live or where results land.

_All stories delivered as of 2026-07-02 (#12–#15); see JOURNAL 2026-07-02 and git. Theme backbone retained per the story-map (themes are the stable backbone; delivered stories/tasks leave per ADR-65)._

---

## [E2] Synthesizer refresh
> As the tool owner, I want the default synthesizer chosen on real scoring data and the choice codified, so the verdict author is evidence-based and cost-aware.
> **Note:** this theme is the baseline gate for the Council-process theme's baseline-gated items.

### [S1] Decide the synthesizer on real data, then codify it
So that the ADR-01 default rests on measured synthesis quality, not assumption.
- [#1] [P1][M] Run the current Gemini synthesizer against ~15 historical transcripts and score with the synthesis rubric · Done when: ~15 transcripts scored + the Phase-3 branch trigger is decided (Branch A swap / Branch B keep) · refs Phase-2 smoke test
- [#2] [P1][M] Implement the Phase-3 conditional: amend ADR-01 (cost-optimization principle) + execute Branch A (new synthesizer) or Branch B (keep Gemini) · Done when: ADR-01 amended + the chosen branch shipped · refs ADR-01 · BLOCKED on #1 (needs the smoke-test scores)
- [#3] [P1][S] Codify the cost-optimization principle in the ADR-01 amendment text (balance quality vs cost) · Done when: the principle is written into the ADR-01 amendment · refs ADR-01 (folded into #2 scope, tracked separately)

### [S2] Settle the panelist/synthesizer overlap policy
So that overlap rules are explicit if the synthesizer ever joins the panel.
- [#4] [P3][S] Amend ADR-02 to codify the cost-reframed panelist/synthesizer overlap policy · Done when: ADR-02 amendment lands (or is closed as not-needed if Gemini retained) · refs ADR-02 · conditional on #2 Branch A

---

## [E3] Provider reliability & CLI engine
> As the tool owner, I want every wired provider to have a known-good, tested path and a route to CLI-subscription backends, so reliability is measured and cost is controllable.

### [S3] Close the untested and unreliable provider paths
So that no provider is wired-but-unverified or silently degrading the panel.
- [#5] [P2][M] Add an integration path for `openai_deep_research.py` (o3-deep-research, ~45 min, ~$10+/run — cannot run in CI) · Done when: a manual integration test path is documented + runnable · refs migrated from tasks/todo.md
- [#6] [P3][M] Evaluate whether DeepSeek should be replaced or demoted from the default full panel · Done when: a replace/keep/demote decision is recorded · refs reactive trigger: round-blocking failure rate >2% per JOURNAL data
- [#20] [P3][S] Fix OpenAI-SDK Responses-API type-stub drift: 6 mypy errors in `src/ai_council/research/providers/{openai_mini,openai_deep,grok}_research.py` (`.create()` overloads, object-not-iterable) · Done when: mypy clean on those files + `check.ps1` fully green · refs pre-existing on main, surfaced 2026-07-02 during Unit 1 merge
- [#21] [P3][S] Fix or delete the stale integration test `tests/test_integration.py::test_full_debate_pipeline` (`ImportError: cannot import name '_build_all_providers' from 'ai_council.cli'` — the symbol no longer exists; the test is `@pytest.mark.integration`, deselected by the canonical unit gate `-m "not integration and not envcheck"`, so it never blocks `check.ps1`) · Done when: the test imports cleanly and passes, or is removed with rationale · refs pre-existing on main, surfaced 2026-07-11 during the #326 arc

### [S4] Add a CLI-subscription provider backend
So that subscription CLIs can serve debate turns and API spend is reserved for CLI-less models.
- [#16] [P3][L] CliProvider engine [Rama 2]: an adapter behind the provider protocol that drives CLI backends (Claude/Gemini/Codex subscriptions); API access reserved for CLI-less providers (DeepSeek/Grok) · Done when: at least one CLI backend runs a debate turn through the provider protocol · refs Rama 2 · design tensions (read-only sandbox, non-determinism, response anonymization, quota-vs-devwork contention) deferred to build-start · do NOT merge provider implementations (keep separate)

### [S5] Cut multi-round input-token cost
So that repeated brief/persona blocks don't re-bill on every provider call and debate round.
- [#128] [P3][S] ai-council prompt caching — apply `cache_control` to the repeated brief/persona blocks reused across providers and debate rounds, cutting input-token cost on multi-round debates · Done when: a council run reuses cached brief/persona blocks and the input-token saving is recorded · refs Anthropic API prompt-caching docs, #96, #110 · re-filed from the hub backlog 2026-07-08 (ADR-41 move, hub commit ea6217a)

---

## [E4] Model currency
> As the tool owner, I want to know when the configured panel models fall behind the latest releases, so the Council never silently debates on stale models.

### [S6] Detect stale model configuration
So that a superseded model in settings.yaml is surfaced, not silently used.
- [#17] [P3][M] Online model-version check: verify the `config/settings.yaml` model strings are the latest available per provider + a documented update process · Done when: a check reports any configured model that is no longer the latest + the refresh process is written down · refs `config/settings.yaml` is the single source of model strings

---

## [E5] Naming & quality automation
> As the tool owner, I want the ADR-34 naming convention enforced mechanically and its edge cases resolved, so violations are caught by CI, not reviewer luck.

### [S7] Enforce hyphen-only naming and resolve its timestamp edge case
So that new files cannot drift from ADR-34 and the ISO-timestamp ambiguity is settled.
- [#7] [P2][M] Add a CI check (pre-commit hook or ruff plugin) rejecting new `docs/`/`src/` files with UPPERCASE or underscores in the slug · Done when: a non-conforming new filename is blocked · refs strażnik finding I5 (SYNTHESIS-QUALITY-RUBRIC.md violation)
- [#8] [P3][S] Decide whether ADR-34 applies to ISO timestamps inside filenames (the `council-out-YYYYMMDD_HHMMSS` underscore) — fix the emitter to hyphens, or amend ADR-34 with an ISO-timestamp exemption · Done when: the methodology decision is recorded + applied · refs ADR-34 · surfaced by #7 once built · adjacent to #14 (same emitter)

---

## [E6] Council process & epistemic quality
> As the tool owner, I want the ADR-67 gated loop implemented, the synthesis rubric sharpened, and the panel's epistemics defended, so the Council runs deterministically and resists framing bias and false consensus.

### [S8] Build the ADR-67 downstream pieces and sharpen the rubric
So that `/council-question` generates + self-gates questions and synthesis quality is unambiguous.
- [#9] [P3][L] Implement ai-council's ADR-67 pieces: the `/council-question` template (one decision + options + constraints + prior-ADR context) and the question-quality gate · Done when: `/council-question` generates a templated question and the gate correctly passes/fails it · refs ADR-67 · DEFERRED — do NOT build before the canonical-baseline settles (mirrors `.dev-knowledge` #70) · NOTE: the deterministic `council.return_dir` I/O clause was moved out to #13 (baseline-INDEPENDENT)
- [#10] [P3][S] Refine the faithfulness criterion in `protocols/SYNTHESIS_QUALITY_RUBRIC.md` to clarify additive meta-analysis cases · Done when: the rubric wording disambiguates synthesizer cross-model synthesis vs raw-transcript content · refs N=1 scoring exercise
- [#11] [P3][S] Provide ai-council's two data points (cycle-1 + cycle-2 retrospective) for the "bilateral handshake = 1 round trip" codification owned by `.dev-knowledge` · Done when: the data points are handed to `.dev-knowledge/LESSONS.md` · refs cross-stream (codification lives in `.dev-knowledge`)

### [S9] Defend the panel's epistemics
So that deadlocks resolve on evidence and consensus is genuine, not a framing artifact.
- [#18] [P3][M] Tool-grounded crux resolution [Rama 1, baseline-gated]: when the panel deadlocks on a factual crux, resolve it by grounding in a tool/evidence lookup rather than more debate · Done when: a factual crux triggers a grounded lookup that feeds the next round · refs Rama 1 · baseline-gated (Synthesizer refresh)
- [#19] [P3][L] Active framing defense + false-consensus alarm [Rama 3, runtime-coupled, baseline-gated]: detect and counter leading/asker-leaked framing at runtime and alarm when apparent consensus is an artifact of framing rather than genuine agreement · Done when: a framing-biased question is flagged + a false-consensus run raises an alarm · refs Rama 3 · runtime-coupled; baseline-gated (Synthesizer refresh)
- [#110] [P3][S] ai-council round-2 isolation audit — verify whether round-2 debaters see round-1 peer arguments; 2026 evidence: 58% sycophantic convergence in debate, 23.9% unanimous-wrong by round 3; confirm blind-vote isolation properties hold for our 2-round design · Done when: isolation properties confirmed or a remediation is recorded · refs #96, ADR-03, research transcript 2026-06-06 · re-filed from the hub backlog 2026-07-08 (ADR-41 move, hub commit ea6217a); ADR-03 is the ai-council blind-voting ADR (ref valid in-repo)

---

**About this file** — ADR-66 story-map (Big Picture → Theme → User Story → Task), migrated
2026-06-02 from the ADR-41/47 stream schema per ADR-38 A6 (canonical backlog form, all
repos). Themes carry a stable `[E<n>]` epic id; stories are human (goal + `So that`) and
carry a stable `[S<n>]` id (#281/#286); tasks carry `[#id] [P][size] · Done when · refs`.
Structure is checked by the `validate-backlog` pre-commit gate (ADR-66). Done tasks **leave** (ADR-65); git is the
implementation record. Conformance is checked read-only by `.dev-knowledge/scripts/audit.py`.

**Grooming log:** 2026-05-12 (stream-format seed) · 2026-06-02 (story-map migration, all 11 items preserved) · 2026-07-02 (6-segment backbone reorganization: 4 themes → 6 lettered segments A–F; #12–#19 added; #9 re-sliced (return_dir I/O → #13); #20 filed under the provider-reliability segment (C) — pre-existing mypy drift surfaced during Unit 1; all 11 prior items preserved; #12 completed + struck (ADR-65) once the protocols/ surface landed — git carries the record) · 2026-07-02 (invocation-surface segment (A) output subsystem shipped: #13 return_dir routing, #14 double-council fix, #15 minority report closed + struck per ADR-65 — commits bfc268f/53ad525/f1a4b74; that segment's story fully delivered, backbone header retained) · 2026-07-08 (Wave-1 onboarding: renamed the thematic backbone from the retired lettered scheme to named themes per ADR-99 clause A; adopted stable `[S<n>]` story ids per #281/#286; re-filed #110 + #128 from the hub backlog per the ADR-41 move (hub commit ea6217a); all task ids preserved) · 2026-07-11 (filed #21 under S3 — stale `test_full_debate_pipeline` integration test surfaced during the #326 arc; next-free local id after #20) · 2026-07-13 (content-parity D1: added `[E1]`–`[E6]` epic ids to the 6 theme headers + the epic-ids backbone line, and wired the ADR-66 `validate-backlog` gate — ADR-78 floor twin, hub audit `2026-07-13-technical-content-parity-inventory.md`; stories/tasks unchanged, all ids preserved). Next quarterly: 2026-10-01.
