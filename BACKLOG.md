# BACKLOG — ai-council

## Big picture

`ai-council` is the ecosystem's multi-model deliberation engine: it runs structured
debate/research across a configurable AI panel and produces the verdicts that become
binding ADRs. The backlog advances the tool toward a delegation-ready, evidence-based,
reliable, and self-enforcing state across seven themes.

**Themes (backbone) — epic ids:** [E1] Invocation surface & delegation-readiness · [E2] Synthesizer refresh ·
[E3] Provider reliability & CLI engine · [E4] Model currency · [E5] Naming & quality automation ·
[E6] Council process & epistemic quality · [E7] Record & governance hygiene

---

## [E1] Invocation surface & delegation-readiness
> As the tool owner, I want ai-council's invocation specs and outputs to live in a clean,
> delegation-ready surface, so an external agent/operator can commission the Council without
> ambiguity about where specs live or where results land.

_Prior stories delivered as of 2026-07-02 (#12–#15); see JOURNAL 2026-07-02 and git. Theme backbone retained per the story-map (themes are the stable backbone; delivered stories/tasks leave per ADR-65)._

### [S10] Deliver the delegation-ready caller surface
So that an external caller gets a contract-honest Lane A and a transcript-free verdict artifact.
- [#22] [P2][S] Close CONTRACT Known-deviation 1: `--file` parses YAML frontmatter via the same `parse_file()` path as inbox (precedence flag > frontmatter > config default) · Done when: a guide-conformant frontmatter brief behaves identically on both lanes, no frontmatter leaks into the question text, and CONTRACT §7 item 1 is removed · refs ADR-11 decision 2, `protocols/COUNCIL_INVOCATION_CONTRACT.md` §7 · post-pause (G2)
- [#23] [P2][S] Close CONTRACT Known-deviation 2: research mode honors `--return-dir` (`run_research` gains a return-dir copy; canonical `./output/` always) · Done when: a Lane A research commission lands its artifacts in the caller's return dir and CONTRACT §7 item 2 is removed · refs ADR-11 decision 3, CONTRACT §7 · post-pause (G2)
- [#33] [P3][S] terra pass-3 re-verification of the #26 verdict-package pass-2 fixes — **explicit-waiver residual** (architect ruling 2026-07-17): pass-3 was not run because codex credits were exhausted (reset 2026-07-23); pass-2 had already confirmed the Critical + all 4 original High resolved, and the two pass-2 fixes are strictly reductive (remove a fabricated minority pointer / remove a manifest overclaim) + unit-tested · Done when: `codex-review.ps1 -Topic verdict-package-pass3` runs on/after 2026-07-23 against the merged verdict-package code and reports no Critical/High (belt-and-suspenders) · refs `docs/audits/2026-07-17-codex-verdict-package-pass2.md`, #26
- [#34] [P3][M] Research-path verdict-package parity (R6 lane-parity): `run_research` (`research/output.py`) emits no `council-verdict-*.json` — only the debate/orchestrator path does. A Lane A research commission therefore gets no transcript-free deliverable · Done when: a research run emits a DRAFT-INT-1 verdict package in every destination, mirroring the debate path · refs L-INT R6, DRAFT-INT-1, the inbox-mirror blind-spot memory; #26 shipped debate-path only per architect ruling 2026-07-17
- [#35] [P3][S] Broad R4 fail-loud return-dir for the transcript + minority artifacts: #26 made only the *verdict* raise `OutputRoutingError` on a required-`--return-dir` write miss; `save_to_file`/`save_minority_report` still treat return-dir failures as best-effort (silent) · Done when: a failed required `--return-dir` write for any Lane A deliverable surfaces loudly (never a bare exit 0), per DRAFT-INT-1 R4 · refs L-INT R4, `output.py` `_write_routed`; scoped-out of #26

### [S13] Stand up the caller-side commissioning advisor ("the window on the world", front half)
So that a delegating agent authors a well-formed brief, splits a compound decision into separate single-decision commissions when warranted, and knows exactly where the outputs are and how to read them — the caller-side ADVISOR that sits in front of the council-side surface [S10] delivered (RIDER 2, session ruling 2026-07-17).
- [#36] [P3][M] Caller-side question-authoring advisor: a template the delegating agent fills to produce a GUIDE-conformant brief (one decision + options + constraints + prior-ADR context) before it invokes Lane A · Done when: an agent can author a conformant `--file` brief from the template without reverse-engineering the GUIDE · refs ADR-11 decision 4 (contract), ADR-67 step-4/6, L-INT §2 (caller journey) · **reconcile with #9** (the ADR-67 `/council-question` template + question-quality *gate* lives in [E6] and is baseline-gated) — this task is the caller-side *authoring surface*, #9 is the *quality gate*; do not duplicate
- [#37] [P3][S] Sub-question decomposition advisory: when a brief bundles more than one decision, advise splitting it into N conformant single-decision Lane A commissions (a caller-side loop the caller controls) · Done when: a compound brief is flagged with a recommended decomposition, each sub-question emerging as a conformant single-decision brief · refs L-INT §3(Q5) — council-side batch stays OUT (one brief, one decision); this is caller-side decomposition, explicitly *not* foreclosed by Q5
- [#38] [P3][S] Outputs read-back guide (verdict→ADR): a documented caller path from the #26 verdict package to a recorded ADR (the F7 verdict→ADR §Q4 section-keyed template) — where the outputs are and how to read them · Done when: a caller can go from `council-verdict-*.json` to a drafted ADR using the template, with the F7 adoption-evidence gate noted · refs L-INT §3(Q4)/F7, DRAFT-INT-1 (#26 delivered the package); template artifact is hub-side, adoption gated on F7 evidence

---

## [E2] Synthesizer refresh
> As the tool owner, I want the default synthesizer chosen on real scoring data and the choice codified, so the verdict author is evidence-based and cost-aware.
> **Note:** this theme is the baseline gate for the Council-process theme's baseline-gated items.

### [S1] Decide the synthesizer on real data, then codify it
So that the ADR-01 default rests on measured synthesis quality, not assumption.
- [#2] [P1][M] Implement the Phase-3 conditional: amend ADR-01 (cost-optimization principle) + execute Branch A (new synthesizer) or Branch B (keep Gemini) · Done when: ADR-01 amended + the chosen branch shipped · refs ADR-01 · Branch A (openai) SHIPPED 2026-07-18 as the durable config default (operator ruling, `docs/audits/2026-07-17-synthesizer-ruling-gemini-to-openai.md`); remaining scope = the ADR-01 amendment text (with #3). Un-gated (Epic B / G3 = the recorded synthesizer ruling; #24 closed by ruling)
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
- [#20] [P3][S] OpenAI/Grok Responses-API type drift — real fix (stopgap applied): the unbounded `openai>=1.50` pin drifted to 2.14.0 (gpt-5.2 rollout), narrowing the `.create()` overloads + tool-param unions and typing annotation containers as `object` → 6 mypy errors in `research/providers/{openai_mini,openai_deep,grok}_research.py`. **Stopgap applied 2026-07-18** (`chore/openai-sdk-mypy-stopgap`): narrow per-line `# type: ignore[call-overload|attr-defined|misc,list-item]` so `check.ps1` mypy is green again; runtime unaffected. · Done when: the real fix lands — migrate the provider calls to the 2.x SDK typing (or set a deliberately bounded `openai` pin) and remove the stopgap ignores · refs `research/providers/*_research.py`, `pyproject.toml` `openai>=1.50` pin; stopgap surfaced during Stream-A #22 (2026-07-18)
- [#21] [P3][S] Fix or delete the stale integration test `tests/test_integration.py::test_full_debate_pipeline` (`ImportError: cannot import name '_build_all_providers' from 'ai_council.cli'` — the symbol no longer exists; the test is `@pytest.mark.integration`, deselected by the canonical unit gate `-m "not integration and not envcheck"`, so it never blocks `check.ps1`) · Done when: the test imports cleanly and passes, or is removed with rationale · refs pre-existing on main, surfaced 2026-07-11 during the #326 arc

### [S4] Add a CLI-subscription provider backend
So that subscription CLIs can serve debate turns and API spend is reserved for CLI-less models.
- [#27] [P3][M] CLI-4 parity run → default-flip decision: n=12 stratified paired debates (CLI vs API), sealed-key blind, rubric-scored, non-inferiority 1/12 with zero margin on items 2 and 4; ratify DRAFT-CLI-3 or retire per its kill condition · Done when: the parity report exists and the flip decision (ratify/retire) is recorded · refs L-CLI §3(Q2)/§4(DRAFT-CLI-3), ADR-12 §5 · prerequisite #16 (CLI seats) delivered 2026-07-17
- [#43] [P3][S] First-class `codex` seat name in the provider registry: `PROVIDER_CLASSES` (`cli.py`) has no `codex` name, so a codex CLI seat must ride a non-openai model name (the night batch hosted it on `deepseek` because `openai` was the synthesizer) — a label/identity mismatch in `seats[]` · Done when: a codex CLI seat can be configured under its own truthful seat name without riding another provider's identity · refs `docs/audits/2026-07-17-night-batch-empirical-e2e-audit.md` §1.2, `cli.py` `PROVIDER_CLASSES`, `seat_router.py` `CLI_PROVIDER_CLASSES`; ADR-12

### [S11] Stand up the liveness surface (doctor)
So that seat/provider health is observable without spend and key hazards cannot silently gate runs.
- [#32] [P3][S] doctor-v2 CLI-fleet auth-lane check — **#16 waived-gate residual** (architect ruling 2026-07-17; terra review finding VERBATIM): *"CLI authentication mode is not verified — Scrubbing environment keys does not guarantee subscription authentication because both CLIs can use API-key credentials stored under the allowlisted home/config directories. A normally configured Codex or Claude CLI previously authenticated with an API key can incur API charges while `metrics.py` records every CLI call as `$0`. Fix direction: Admit the CLI lane only after verifying subscription/OAuth authentication; otherwise fall back to API accounting or reject the seat."* · Done when: the doctor (or seat-build) verifies each CLI seat's auth lane (subscription vs API-key) and the `$0` CLI-cost recording is gated on witnessed subscription auth (else API accounting) · refs the documented assumption at `src/ai_council/metrics.py` `build_call_metrics` (which names this item back); ADR-12 §Decision 3 CLI-fleet; doctor-v2

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
- [#42] [P3][S] Fix the research-path double-prefix filename: `research/output.py` hardcodes the `research-` mode token, so a slug that also begins "research" yields `council-out-<ts>-research-research-<slug>.md` · Done when: a research run whose query begins "research…" produces a single-prefixed filename (strip a leading `research-` from the slug, or route research through the shared mode formatter) · refs `docs/audits/2026-07-17-night-batch-empirical-e2e-audit.md` §3.1 (H1), `research/output.py:41`; adjacent to #8 (naming)

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
- [#18] [P3][M] Tool-grounded crux resolution [Rama 1]: when the panel deadlocks on a factual crux, resolve it by grounding in a tool/evidence lookup rather than more debate · Done when: a factual crux triggers a grounded lookup that feeds the next round · refs Rama 1, DRAFT-EPI-3 · **un-gated 2026-07-18** (Epic B / G3 event); planning deferred to a future session. Night council (2026-07-17) recommended option (c): a bounded crux-check step between rounds — see `docs/audits/2026-07-17-night-batch-empirical-e2e-audit.md` §2.2
- [#19] [P3][L] Active framing defense + false-consensus alarm [Rama 3, runtime-coupled]: detect and counter leading/asker-leaked framing at runtime and alarm when apparent consensus is an artifact of framing rather than genuine agreement · Done when: a framing-biased question is flagged + a false-consensus run raises an alarm · refs Rama 3, DRAFT-EPI-4 · runtime-coupled; **un-gated 2026-07-18** (Epic B / G3); planning deferred. Night council recommended option (b): a debate-time framing-challenge role, shadow-mode first — see night audit §2.2
- [#110] [P3][S] ai-council round-2 isolation audit — verify whether round-2 debaters see round-1 peer arguments; 2026 evidence: 58% sycophantic convergence in debate, 23.9% unanimous-wrong by round 3; confirm blind-vote isolation properties hold for our 2-round design · Done when: isolation properties confirmed or a remediation is recorded · refs #96, ADR-03, research transcript 2026-06-06 · re-filed from the hub backlog 2026-07-08 (ADR-41 move, hub commit ea6217a); ADR-03 is the ai-council blind-voting ADR (ref valid in-repo)

---

## [E7] Record & governance hygiene
> As the tool owner, I want the decision record current and its lifecycle conventions ratified, so the canonical record never drifts from live state.

_[S12] Execute the GOV-1 currency pass — **delivered 2026-07-17** (#31: 15 lane-doc rulings → RULED, ADR-09/10 → Accepted, DRAFT-GOV-1 ratified as ADR-14, feature-work pause lifted; see JOURNAL + `docs/intake/2026-07-17-gov1-rulings-register.md`). Theme backbone retained per the story-map._

---

**About this file** — ADR-66 story-map (Big Picture → Theme → User Story → Task), migrated
2026-06-02 from the ADR-41/47 stream schema per ADR-38 A6 (canonical backlog form, all
repos). Themes carry a stable `[E<n>]` epic id; stories are human (goal + `So that`) and
carry a stable `[S<n>]` id (#281/#286); tasks carry `[#id] [P][size] · Done when · refs`.
Structure is checked by the `validate-backlog` pre-commit gate (ADR-66). Done tasks **leave** (ADR-65); git is the
implementation record. Conformance is checked read-only by `.dev-knowledge/scripts/audit.py`.

**Grooming log:** 2026-05-12 (stream-format seed) · 2026-06-02 (story-map migration, all 11 items preserved) · 2026-07-02 (6-segment backbone reorganization: 4 themes → 6 lettered segments A–F; #12–#19 added; #9 re-sliced (return_dir I/O → #13); #20 filed under the provider-reliability segment (C) — pre-existing mypy drift surfaced during Unit 1; all 11 prior items preserved; #12 completed + struck (ADR-65) once the protocols/ surface landed — git carries the record) · 2026-07-02 (invocation-surface segment (A) output subsystem shipped: #13 return_dir routing, #14 double-council fix, #15 minority report closed + struck per ADR-65 — commits bfc268f/53ad525/f1a4b74; that segment's story fully delivered, backbone header retained) · 2026-07-08 (Wave-1 onboarding: renamed the thematic backbone from the retired lettered scheme to named themes per ADR-99 clause A; adopted stable `[S<n>]` story ids per #281/#286; re-filed #110 + #128 from the hub backlog per the ADR-41 move (hub commit ea6217a); all task ids preserved) · 2026-07-11 (filed #21 under S3 — stale `test_full_debate_pipeline` integration test surfaced during the #326 arc; next-free local id after #20) · 2026-07-13 (content-parity D1: added `[E1]`–`[E6]` epic ids to the 6 theme headers + the epic-ids backbone line, and wired the ADR-66 `validate-backlog` gate — ADR-78 floor twin, hub audit `2026-07-13-technical-content-parity-inventory.md`; stories/tasks unchanged, all ids preserved) · 2026-07-16 (plan-of-record reconciliation: filed #22–#31; new theme [E7] + stories [S10]–[S12] — additive, all prior ids preserved; #16 gained an ADR-12/pre-work clause; the phase→task map lives in `docs/intake/2026-07-16-plan-of-record.md`) · 2026-07-17 (GOV-1 consolidation, gate G1→G2: 15 lane-doc §6 rulings recorded → RULED in `docs/intake/2026-07-17-gov1-rulings-register.md`; **FORK_RULING(a)** — #1 absorbed into #24 (#24 is the authoritative full-corpus evidence path for #2's Branch A/B decision; #2 re-pointed #1→#24); ADR-09/10 → Accepted + DRAFT-GOV-1 ratified as ADR-14; feature-work pause lifted). · 2026-07-17 (**#28 CLOSED** — F3 grok cost lane: operator ran `grok login` (subscription OAuth); repo re-probe with scoped env-key shielding witnessed the subscription lane ("logged in with grok.com"; `~/.grok/auth.json` present) — done-when met, disposition amended in `docs/audits/2026-07-17-f3-grok-cost-lane-disposition.md`; grok seat still deferred to #27 parity, pins grok-4.5, scope-shields `XAI_API_KEY` per call). · 2026-07-17 (**#16 CLOSED** — CLI seats claude+codex: A1 template-method base + A3 one-classifier/retry (MERGE 1) then CLI adapters + seat-router admission gate + same-seat API fallback + `seats[]` sidecar + backend axis (MERGE 2); witnessed live — both CLI seats participate end-to-end + harness-induced `fallback_events[cause=process-error]`; CLI calls $0-cost. **Filed #32** (doctor-v2 CLI-fleet auth-lane check) as the explicitly-waived terra-gate residual per architect ruling). Next quarterly: 2026-10-01. · 2026-07-17 (**#25 struck** — `council doctor` v1 delivered at merge `6e0782e` (DRAFT-DOC-1 + A2 `@click.group`; terra-clean); done-task leaves per ADR-65, git carries the record. Pre-step of the #26 verdict-package arc; [S11] retains open #32.) · 2026-07-17 (**#26 CLOSED** — verdict package per DRAFT-INT-1: `save_verdict_package` sibling of `save_to_file` (pre-work A4 decompose + B3 `_ts()` helper) emitting `council-verdict-<ts>-<mode>-<slug>.json` to every destination with the full 14-field set; human-readable mirror block folded into `_build_header`; consumes `seats[]`/`synthesis` by reference via the shared `_seat_payload`. Witnessed live on shipping code (merge below): schema-valid package, deterministic `<ts>`, transcript-free property demonstrated. Architect rulings recorded: contract_version=null, exit_semantics=0, per-item extraction/record annotations, mirror block via the amended save_to_file contract. **Filed #33** (terra pass-3 waiver residual — codex credits exhausted, reset 2026-07-23), **#34** (research-path parity, R6), **#35** (broad R4 fail-loud). Terra pass-1/pass-2 clean-on-substance after fixes; pass-3 EXPLICITLY WAIVED per architect ruling.) · 2026-07-17 (**RIDER 2** — session ruling, filing only: coverage of the operator's delegation-window ADVISOR layer (caller-side authoring → sub-question decomposition → outputs read-back) checked against ADR-11 + [E1] = PARTIAL; [S10] delivered the council-side surface but the caller-side front half was uncovered. Filed **new story [S13]** with **#36** (authoring advisor; reconcile w/ #9 quality gate), **#37** (decomposition advisory; caller-side, Q5-compatible), **#38** (verdict→ADR read-back guide, F7). Zero build, zero new folders.) · 2026-07-18 (**morning close** of the 2026-07-17 night batch: **#31 struck** — [S12] delivered 2026-07-17, collapsed to a theme-level note; **#24 CLOSED by operator ruling** — the G3/Epic-B event is the recorded synthesizer ruling (`docs/audits/2026-07-17-synthesizer-ruling-gemini-to-openai.md`), the EPI-1 40-item pack retained as the reversible instrument; **Epic B un-gated** — #2 Branch A (openai) shipped as the durable config default (ADR-01 amendment text pending), #18/#19 un-gated with planning deferred to a future session; filed **#39–#43** from the night-batch E2E audit — `--no-persist` output guard, `options_considered` extractor, CLI-seat token regexes, research double-prefix, first-class codex seat name.) · 2026-07-18 (**#40 + #41 struck** — Stream B `worktree-p6-hardening` integrated to main via `--no-ff` merge `edce0e7`: #40 verdict-package `options_considered` extractor (`496a2c9`) + #41 CLI-seat token counts (`8a5cecd`); both E2E-witnessed + contract-green, primary `check.ps1` 531 passed; done-task leaves per ADR-65, git carries the record. terra WAIVED to 2026-07-23 — #40/#41 go first in the batch re-review.) · 2026-07-18 (**#39 struck** — serial build in primary after Stream B integration: `council --no-persist` (scratch temp) + `AICOUNCIL_OUTPUT_DIR` env override + bounded `output/health/` retention (keep-10 prune in `doctor.write_record`; documented in the doctor module docstring, ops home outside the CONTRACT — witness/dev controls are not the Lane-A delegation surface, so Contract-Version stays 1.0 and #34 keeps the 1.1 earmark). check.ps1 537 passed; terra WAIVED to 2026-07-23. Done-task leaves per ADR-65.)
