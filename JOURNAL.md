# Journal — ai-council

<!-- scope: meta -->

> Per-session tactical log of `ai-council` Claude Code work. Entry shape
> as of 2026-05-16 (Council Simplification): `Did / Result / Changes /
> Abandoned / Next`. Newest-first prepend ordering.
>
> Distinct from LESSONS (per-learning generalized rules, newest-first per
> ADR-29) and handoffs (per-session boundary artifacts for browser-chat
> resumption). JOURNAL is the within-Claude-Code-sessions tactical log
> enabling context recovery across sessions in same repo. The `Changes:`
> line records what files / areas moved — replacing the deleted CHANGELOG.md.
>
> Update protocol: prepend new session entry at top of entry list (under this
> intro blockquote, before existing entries). One entry per Claude Code session
> OR per workday for heavy days. Each entry cites commit hashes, handoff doc,
> or ADR for deeper detail. JOURNAL summarizes, doesn't duplicate.

---

### 2026-07-17 — GOV-1 consolidation (#31, gate G1→G2): feature-work pause LIFTED

**Did:** Day-session **Beat 2** — executed #31 (GOV-1 currency pass) MINUS the two EPI-1-dependent parts (Beat 1 deferred — see Not-done). **(1)** Recorded all **15** lane-doc §6 rulings verbatim → RULED + the #1/#24 **FORK_RULING(a)** in `docs/intake/2026-07-17-gov1-rulings-register.md` (acceptance scope = ACCEPT ALL over the §6 primary set; supersedes the intake §5 label drift) — `ec406a4`. **(2)** Flipped **ADR-09/10 → Accepted** (header + `docs/decisions/README.md` index, same commit per the sync invariant) and ratified **DRAFT-GOV-1 → ADR-14** (ADR-13 stays reserved for the crux resolver) — `85a692f`. **(3)** CLAUDE.md §11 extended the local ADR list ADR-08 → **ADR-09/10/11/12+14**; genuine re-read (only §11 was stale), re-stamped 2026-07-17 — `8bde22d`. **(4)** VISION:25 reconciled the always-on dual-write claim → ADR-43/ADR-10 (local default + opt-in mirror), re-stamped — `2ae0b99`. **(5)** CONTRIBUTING re-read **caught + fixed real staleness** (its ADR status-values line contradicted the just-ratified ADR-14 lifecycle), re-stamped — `4629d63`. **(6)** BACKLOG absorbed **#1 into #24** per FORK_RULING(a) (#24 = authoritative full-corpus evidence path for #2; #2 re-pointed #1→#24; grooming-log noted) — `8ed4012`.

**Result:** **Feature-work pause LIFTED (O2)** — gate G1 (operator acceptance of the 15 rulings) closed, G2 (GOV-1 execution) satisfied. Pre-merge gate: **unit suite 444 passed, 6 deselected (unchanged vs main)**; mypy shows only the **6 pre-existing #20** Responses-API stub-drift errors (zero `.py` changed this arc — byte-identical, no new errors); ruff unaffected. ADR-09/10 Accepted in header AND index; ADR-14 exists + indexed; `validate-backlog` OK; `canonical_freshness` satisfied by genuine re-stamps (CLAUDE/VISION/CONTRIBUTING all 2026-07-17). Zero `src/` changes.

**Changes:** `docs/intake/2026-07-17-gov1-rulings-register.md` (new), `docs/decisions/{ADR-09,ADR-10}` (Accepted), `docs/decisions/ADR-14-adr-lifecycle-states.md` (new), `docs/decisions/README.md` (index), `CLAUDE.md` (§11+stamp), `VISION.md` (:25+stamp), `CONTRIBUTING.md` (ADR status-values+stamp), `BACKLOG.md` (#1 absorbed), `JOURNAL.md` (this entry). Session SHAs: `ec406a4`, `85a692f`, `8bde22d`, `2ae0b99`, `4629d63`, `8ed4012`.

**Abandoned:** none.

**Not done (deferred):** **Beat 1 (EPI-1 report + ruling)** — the operator's blind scoring is not yet done (0/40); per ruling r3 CC cannot score. Beat 1 re-arms as its own mini-session once scoring completes; the EPI-1 ruling then enters the rulings register as an **addendum** and un-gates Epic-B (#18/#19/#9 + v2 resolver). Epic-B items are NOT un-gated and NOT started. DRAFT-GOV-2 not ratified (out of scope).

**Next:** push `main` at close-out (ruling 15; this arc's merge SHA anchored in the follow-up below); Beat 3 (Leg C #29/#30) unlocks post-pause; EPI-1 workspace relocation (housekeeping).

### 2026-07-16 — EPI-1 archaeology blind-scoring pack PREP (Leg A) — #24 prep, NOT #24 close

**Did:** Overnight autonomous PREP for BACKLOG **#24** (EPI-1 archaeology); method canon `docs/intake/2026-07-06-lane-epi-functional-design.md` §3(Q3); operator rulings applied — **r1** comparative gemini-vs-openai (OQ-1), **r3** operator scores blind + LLM-judge = second opinion only (OQ-3), **r5** corpus = full local `output/` + hub dedupe, no curation (FE-1). **Read-only** mined the full 239-file `output/` corpus → **138** identity-readable syntheses (openai 56 / claude 54 / gemini 21 / claude-sonnet 7, incl. 3 openai-`participant` + 1 pre-label anomalies — reconciles **exactly** with the lane doc's witnessed tally). Segmented by de-facto verdict author; built a **blind scoring pack** of **40** items (**20 gemini + 20 openai**, both non-participant), matched on mode (decision), panel size (4-model), and era (by month `{03:4,04:3,05:3,06:10}`); excluded 3 research-mode + 1 FAILED run (documented, not curation-for-outcome). Anonymized (Synthesizer/Panel/header redacted), relabeled + shuffled (seed 20260716). Ran a 5-way **blind** Sonnet **LLM-judge second-opinion** pass (judges had no key).

**Result:** Pack self-sufficient at `output/epi1-archaeology/` (OPERATOR-SCORING-README · CORPUS-MANIFEST · scoring-sheet · items/ITEM-01..40). Identity key **SEALED OUTSIDE** the pack at `output/epi1-archaeology-KEY-SEALED.json`. LLM-judge report (labeled `second-opinion`, **segregated**) at `output/epi1-archaeology-SECOND-OPINION-judge.md`. All pack artifacts are **gitignored** under `output/` (anonymized transcripts + sealed key + judge scores must not be committed) — this JOURNAL entry is the sole tracked artifact. Hub dedupe: **37 of 41** timestamped hub files are confirmed local mirrors (deduped); 4 hub-only (2 research + 2 pick 2026-05-15) excluded per r5. Judge second-opinion: **gemini ≈ openai** (95% overall each; C1–C3/C5 perfect, one C4/faithfulness slip apiece: ITEM-39/gemini, ITEM-33/openai) — explicitly NON-verdict. `output/` corpus **byte-untouched** (read-only reads only; working tree clean; top-level `output/*.md` count 168 unchanged); zero `src/` edits.

**Changes:** `JOURNAL.md` (this entry). Gitignored (NOT tracked): `output/epi1-archaeology/**`, `output/epi1-archaeology-KEY-SEALED.json`, `output/epi1-archaeology-SECOND-OPINION-judge.md`. Branch `docs/epi1-archaeology-prep`; work commit `a6ba6ce` (Leg A; sole tracked artifact).

**Abandoned:** none.

**Next (operator morning event — NOT done here):** #24's own done-when stays **OPEN** — the single-recommendation report (Branch A swap / Branch B keep) + the operator ruling is tomorrow's operator event; it was **NOT** pre-drafted. Operator runbook: open `output/epi1-archaeology/OPERATOR-SCORING-README.md`, score `items/ITEM-01..40.md` into `scoring-sheet.md` (5 criteria, blind), then un-blind with `output/epi1-archaeology-KEY-SEALED.json` only after all 40 are scored.

### 2026-07-16 — plan-of-record arc anchor: merge `6a5e595`

**Did/Result:** anchors the plan-of-record arc merge SHA `6a5e595` (--no-ff, pushed) — the citation the plan doc's §4 P0 row points to (amended closure: "P0 maps to this arc's plan doc cited by merge SHA"). Work SHAs in the entry below.

### 2026-07-16 — plan-of-record arc: pointer sweep + BACKLOG reconciliation (#22–#31, [E7]) + `docs/intake/2026-07-16-plan-of-record.md`

**Did:** Docs-only arc on `docs/plan-of-record` (plan mode; operator amended the closure in-session: phase rows map to a task ID **or** a named session-event/gate; GOV-1 ruled into new theme [E7]). **(1)** Pointer sweep — fixed the five stale Reading-Map lane-doc paths in the technical-architect intake left by the `b849653` relocation (`a71a5fe`); JOURNAL history + immutable audits untouched. **(2)** BACKLOG reconciliation at task grain — filed #22–#31 (CONTRACT §7 deviation closures, EPI-1 archaeology, doctor v1, verdict package, CLI-4 parity→flip, F3 grok OAuth, F12 stale pin, DOC-3 secrets rule, GOV-1 execution) under new stories [S10]–[S12] + new theme [E7]; #16 gained the ADR-12 v1=claude+codex scope + A1→A3 pre-work note (`ab08ab0`). **(3)** Materialized the plan-of-record (frozen P0–P6 phase plan, gates G1–G3, seam rules, phase→task map, P4 pre-work map) in the intake layer (`320bd70`). **(4)** Codex doc-lane review (terra) returned 2 HIGH + 4 MEDIUM — all addressed (`d12abe5`: #16 v1-set done-when, P6 completion-backstop note, #31 gains the #1/#24 reconciliation, seven-theme counts, P0 citation location).

**Result:** `validate_backlog` OK (7 themes, 12 stories, 29 tasks, 0 warnings); zero editable stale lane-doc pointers (grep-verified); every phase row mapped per the amended closure. ADR-09/10 untouched (`Proposed` — flip is #31's, consolidation session). Zero `src/` changes.

**Changes:** `BACKLOG.md` (+#22–#31, [E7]/[S10]–[S12], grooming log), `docs/intake/2026-07-06-technical-architect-intake.md` (Reading-Map paths + backlog row; appendix deltas 7–8), `docs/intake/2026-07-16-plan-of-record.md` (new), `docs/audits/2026-07-16-codex-plan-of-record.md` (new, review artifact), `JOURNAL.md` (this entry). Session SHAs: `a71a5fe`, `ab08ab0`, `320bd70`, `d12abe5`.

**Abandoned:** none. `docs/audits/2026-07-06-code-quality-audit.md:5` bare-filename mention deliberately left (immutable audit; documented in the plan doc §6).

**Next:** G1 — operator acceptance of the 15-item rulings register → consolidation session executes #31 (GOV-1) → pause lifts; #24 (EPI-1 archaeology) is pause-independent and runnable now.

### 2026-07-16 — fleet_parity ARC-B legs 1+2: DECLARE dep-pytest-xdist + .vscode (fleet #328)

**Did:** Machine-declared ai-council's two known-good divergences in `.methodology.yaml` (branch `chore/methodology-declares-arcb`, work commit `f62cd80`) — **(1)** `dep-pytest-xdist` (pytest-xdist 3.8.0 in `.venv`, unpinned in pyproject → the #328 dep leg read "installed but UNDECLARED"; DECLARED not pinned per operator ruling; shelf-life 2026-08-16 to revisit pin-vs-ambient); **(2)** `.vscode` (shared editor settings, register-e1 short shelf-life 2026-08-13). **Declarations ONLY** — no pyproject/behavioral change. Terra reviewed clean. Merged by CC-primary on the operator's GO (`--no-ff`).

**Result:** both ai rows now **PASS-declared** in `fleet_parity` (were WARN-undeclared). Canonical unit gate GREEN — `pytest -m "not integration and not envcheck"` = 444 passed, 6 deselected (the #21 integration failure is unchanged/deselected).

**Changes:** `.methodology.yaml` (+2 declares), `JOURNAL.md` (this entry). Work commit `f62cd80`.

**Abandoned:** none (ARC-A leg-1 hub-block split untouched — hub `[#336]`).

**Next:** ARC-B completes after the corp `.vscode` declare (leg 3).

---

### 2026-07-16 — fleet_parity ARC-A leg 3b: .gitignore .hypothesis/ parity (fleet #328)

**Did:** Added `.hypothesis/` to root `.gitignore` (branch `chore/gitignore-hypothesis-parity`, work commit `03886c0`), bringing ai-council to fleet IGNORE-tier parity — the #328 `fleet_parity` checker's `ignore-hypothesis` row was WARN-undeclared (ai lacked the line; `*.egg-info/` was already ignored). Effect probe verified: `git check-ignore -q .hypothesis/x` now exits 0 → AT-PARITY (no `.methodology.yaml` declaration needed). Terra (gpt-5.6-terra) reviewed clean. Merged to main by CC-primary on the operator's GO (merge `5eb15b9`, `--no-ff`; merge execution delegated, operator = authorization gate).

**Result:** canonical unit gate GREEN — `pytest -m "not integration and not envcheck"` = **444 passed, 6 deselected** (unchanged vs main). A raw full `pytest -q` shows 1 failure — the deselected-by-design integration test `test_full_debate_pipeline` (`ImportError: _build_all_providers`), already tracked by **#21**; not introduced here, not a canonical-gate failure. `fleet_parity` against the three mains = 161 at-parity / 4 warn-undeclared / 0 must-absent.

**Changes:** `.gitignore` (+`.hypothesis/`), `JOURNAL.md` (this entry). Work commit `03886c0`; merge `5eb15b9`.

**Abandoned:** none.

**Next:** ARC-B DECLARE leg — declare ai `dep-pytest-xdist` (installed 3.8.0, undeclared) + `.vscode` in `.methodology.yaml`.

---

### 2026-07-13 — W3 AI legs micro-arc: .gitattributes parity verify + root .env deletion (commit-and-STOP)

**Did:** Three bounded legs on `chore/w3-ai-legs`. **(1) `.gitattributes` byte-parity** with the ruled fleet baseline (`* text=auto eol=lf` + `*.ps1 text eol=crlf`; ai-council is the **source form**): verified **no deviation** — both ruled directives present verbatim (file is 196 bytes, LF, UTF-8 for the `#282` em-dash comment), so **no adjustment made**. `git add --renormalize --dry-run .` listed all **146** tracked files, but that is dry-run *processing* noise, not blob churn — measured **real** churn non-destructively via a throwaway `GIT_INDEX_FILE` temp-index copy: `git diff --cached --numstat HEAD` on the renormalized temp index = **0 files changed**. Tree already fully LF-normalized per #282. Real index **never touched**; did **NOT** renormalize. **(2) Root `.env` DELETED** (operator verb-ruling): confirmed it was **two comment lines, no values** (`# API keys loaded from Documents/.secrets/.env via PowerShell profile` / `# No local overrides needed`; 100 bytes, CRLF) and **untracked + ignored** (`git check-ignore .env` exit 0, `git ls-files .env` empty) → plain `rm`, verified absent. Real key source is the global `~/Documents/.secrets/.env` per the file's own comment; deleting the empty CWD-fallback stub changes no runtime behavior (`cli.py` `load_dotenv(override=False)` — global wins regardless). Being untracked+ignored, deletion leaves the working tree **clean** (no git surface). **(3)** This JOURNAL entry with the embedded session-summary block below.

**Result:** Gates green — `ruff check src/ tests/ scripts/` **All checks passed**; unit suite **444 passed** (6 deselected). Zero `.py` / config changes in-arc (only this markdown entry + an untracked-file deletion), so mypy is byte-identical to `main` — the 6 pre-existing BACKLOG #20 Responses-API stub-drift errors are unchanged, not a regression. **Merge HELD** — commit-and-STOP; the merge is the operator's from this repo (do not `/ship`).

**Changes:** deleted untracked `./.env` (not a tracked-file change — no diff); `JOURNAL.md` (this entry). Branch `chore/w3-ai-legs`; work commit `5e7ff1a` (this entry is the arc's **only tracked artifact** — `.gitattributes` verified-unchanged, `.env` untracked so no diff) + this anchor follow-up. `.gitattributes` **inspected, unchanged** (parity confirmed).

**Abandoned:** nothing.

**Next:** operator merges `chore/w3-ai-legs` at will.

<!-- session-summary (hub handoff generator reads this) -->
```
DECISIONS:
  - .gitattributes matches the ruled fleet baseline byte-for-byte (source form) — no edit.
  - Renormalize NOT run: real blob churn = 0 (temp-index measurement); the 146-file
    dry-run listing is processing noise, not churn. Do-not-renormalize instruction honored.
  - Root .env deleted per operator verb-ruling (empty stub, untracked+ignored, safe plain rm).
CHANGES:
  - deleted: ./.env (untracked+ignored; 2 comment lines, 0 values; global secrets unaffected)
  - JOURNAL.md: this entry
  - .gitattributes: verified, UNCHANGED
PENDING:
  - Operator to merge chore/w3-ai-legs (--no-ff); merge HELD this session.
  - Pre-existing (not this arc): BACKLOG #20/#21 — 6 mypy Responses-API stub-drift errors.
CONTEXT:
  - Branch chore/w3-ai-legs off main; work commit 5e7ff1a. Gates: ruff clean, pytest 444 passed.
  - No tracked-file content changed except JOURNAL.md; .env removal is untracked → tree clean.
```

---

### 2026-07-12 — Witnessed operational-minimum verification (A0 exit leg b, intake #13 v4; report-only, commit-and-STOP)

**Did:** Read-mostly witnessed verification of every organ class against the `f8f9e58` content-parity baseline (the only mutations permitted were throwaway probes, all created+deleted in-arc, and this JOURNAL entry). For each organ, produced EXECUTED evidence, not a listing. **HOOKS** — pre-commit: `ruff` (probe with unused `import os` → Failed exit 1 F401/I001; clean probe → Passed exit 0), `validate-audit-casing` (staged `docs/audits/BADCASE-Probe.md` → refused exit 1 ADR-101 R4; staged lowercase-kebab → exit 0), `validate-backlog` (clean → OK exit 0, 6 themes/9 stories/19 tasks; appended task missing "Done when:" → FAIL exit 1; BACKLOG.md restored). commit-msg: `backlog-id-on-close` (staged removal of task `[#4]`, message without id → exit 1; message with `[#4]` → exit 0; restored). pre-push: `block-ff-push` verified armed (`.git/hooks/pre-push` installed+wired, config-listed; `pre-commit run --hook-stage pre-push` Passed on empty real range; synthetic native-stdin push-to-main with all-zeros remote → REFUSED exit 1 naming 11 non-merge first-parent commits — NO push). SessionStart: `check_floor_hash.py --require-present` exit 0 + `pre-commit install` exit 0. Stop: `session_end_backpressure.py` fired its advisory dirty-tree leg while probes were on disk (fire-once mechanism witnessed) then went SILENT exit 0 on a clean tree — the hard leg is a no-op at `base..HEAD = 0` and clean-tree silence is DESIGNED (2026-07-12 diagnosis). **SKILLS** — `gotchas` (user) launched + loaded; no repo `.claude/skills/` (correctly absent per CLAUDE §8). **COMMANDS** — `/override` dry reject-on-empty-reason ("Nothing armed"; no token/log written); `/session-summary` preflight side-effect-free (TOKEN-LOG last entry 2026-07-09, 3d ≤ 7 → skips ccusage/staging); `/ship` intentionally not run. **HANDOFF** — ADR-36/42 posture matches CLAUDE §1 verbatim (no local `docs/handoffs/`; hub `../.dev-knowledge/docs/handoffs/` present); no generation attempted.

**Result:** PASS/FAIL table = **14 witnessed / 13 PASS / 0 FAIL / 1 SKIPPED-by-design** (`/ship`, no merge in arc). No FAIL → the P1 stop-for-fix path was not triggered. All probes deleted; tree returned clean; HEAD unchanged through the witness phase (`f8f9e58`).

**Changes:** none persisted beyond this entry — the arc mutated no tracked state (all evidence via ephemeral probes). This is the report-only close-out; committed on `chore/witnessed-opmin-verify` and **held (not merged)** per commit-and-STOP.

**Abandoned:** nothing.

**Next:** operator merges `chore/witnessed-opmin-verify` at will (do NOT `/ship` from here). Gate note: this is a journal-only commit — it cannot cite a work SHA because the arc shipped none, so the ADR-85 hard gate was cleared via `/override` (reason: known journal-only gate edge; same disposition as corp `9be4c35`), HEAD-bound and logged to `logs/OVERRIDES.md`.

---

### 2026-07-13 — Content-parity T1 canonical baseline (consumer half; commit-and-STOP, merge HELD)

**Did:** Applied the operator-ruled consumer half of the fleet content-parity rollout (authority: hub audit `2026-07-13-technical-content-parity-inventory.md` @ `2b21cb26`, byte-verified unchanged at hub HEAD; templates `../.dev-knowledge/templates/claude-regions/` + `{CONTRIBUTING,JOURNAL,LESSONS}-md-template.md`). **(1) CLAUDE.md §1/§4/§5/§6/§10** — the 8 `owner=hub` regions expanded **4 → 8** and materialized **byte-verbatim** (checker-verified all 8 match their templates incl. leading blank lines): `first-read` (A1, +adjacent repo handoff note), `conventions-commit-branch` (A2), **+`conventions-output-formatting`** (A3), `critical-rules-records` (A4, canon 3 items), **+`critical-rules-consistency`** (A5, slot 6), **+`critical-rules-no-leftovers`** (A6, slot 9), `session-start-protocol` (A7, +adjacent repo merge-note), **+`antipatterns-universal`** (A8). §5 renumbered to the hub **1–10 skeleton** (hub regions 1-3/6/9 — the verbatim "6."/"9." bodies force the skeleton; all 5 local rules preserved at 4/5/7/8/10). B2 wrapped contiguous §4/§5 project prose `owner=repo`; B4 §8 relabels the three `.claude/rules/` as **rules, not skills**; B5 §9 gains the two missing live local ids (`validate-audit-casing`, `validate-backlog`). **(2) CONTRIBUTING.md** rebuilt on the template shell — CANONICAL Branch naming / Commit style / Backlog-id (incl. the D2 `hub#`/`ai#`/`corp#` rule) / Definition of done byte-verbatim (anchor-verified); LOCAL shell-quoting caveat + Council ADR-43 routing + Validators roster set-matched; `reconciled_with: handoff-process@5.7`. **(3) LESSONS/JOURNAL** adopted the canonical preambles (JOURNAL had none) — every existing entry grandfathered untouched (append-only, verified no `### 20xx` entry changed). **(4) BACKLOG.md** D1 — 6 theme headers carry `[E1]`–`[E6]` + the epic-ids backbone line; stories/tasks unchanged. **(5) validate_backlog** wired — the ADR-78 **floor twin** copied byte-identical to `scripts/validate_backlog.py` + a `validate-backlog` local pre-commit hook (path resolves in-repo; passes). **(6) .gitignore F1** — hub `.claude` track-by-default policy (ignore only `settings.local.json` + `worktrees/`); **E3** `git rm --cached .claude/settings.local.json` (untrack only, file on disk). **(7) .methodology.yaml** — new `claude-md-section-11-title` divergence: §11 keeps its title **without** the hub "(last 5)" suffix (operator ruling — a "(last 5)" label over a ~20-item curated list is a false claim; label accuracy > cosmetic parity). F3 (`protocols/` genre split) verified already-conformant — no change.

**Result:** `.\scripts\check.ps1` — **pytest 444 passed** (6 deselected), **ruff clean** (full tree src/tests/scripts). `check.ps1` exits 1 at **mypy** on the **6 pre-existing** Responses-API stub-drift errors (BACKLOG #20/#21) — this arc touched **no `src/` .py** (only added `scripts/validate_backlog.py`, outside mypy's `src/` scope), so mypy is byte-identical to `main`, **not a regression**. `validate-backlog` gate **passed live** in the BACKLOG commit's pre-commit run. **REVIEW:** codex doc-review (default model, full branch diff) → **CLEAN disposition-faithfulness + CLEAN no-hub-only-leaks**; one COSMETIC (CONTRIBUTING claimed hub hooks from `repo: ../.dev-knowledge` vs the real GitHub URL) **fixed in-branch** (`7239476`). codex code-review (validate_backlog.py + config subset) → **CLEAN** (even ran the validator under cp1252 mode — no Windows encoding crash). **Merge HELD** — commit-and-STOP; merge is the operator's from this repo (do not `/ship`).

**Changes:** `CLAUDE.md` (§1/§4/§5/§6/§8/§9/§10/§12 + `last_reviewed` 2026-07-13), `CONTRIBUTING.md` (template shell), `LESSONS.md` + `JOURNAL.md` (preambles), `BACKLOG.md` (`[E<n>]` spine), `.pre-commit-config.yaml` + `scripts/validate_backlog.py` (new gate), `.gitignore`, `.methodology.yaml` (§11 divergence), `.claude/settings.local.json` (untracked). Branch `chore/content-parity-t1-baseline`; commits `649b484` (validate_backlog) · `b46e6ce` (CLAUDE 8 regions) · `0887391` (CONTRIBUTING) · `db24af6` (LESSONS/JOURNAL) · `d420399` (BACKLOG E-spine) · `8361660` (gitignore F1/E3) · `d96ef6d` (methodology §11) · `7239476` (doc-review fix) + this JOURNAL commit. **Not merged — held for operator review.** Caveat: `scripts/validate_backlog.py` is a hand-maintained ADR-78 twin with no automated carrier yet (intake G1/G2) — re-sync by hand on hub changes.

---

### 2026-07-12 — Fleet ruling: ruff-gate RE-ACTIVATION + hub `.vscode/settings.json` carry (commit-and-STOP; push HELD)

**Did:** Executed two fleet rulings. **(1) Ruff pre-commit gate RE-ACTIVATED** — wired the `astral-sh/ruff-pre-commit` mirror into `.pre-commit-config.yaml`, pinned **`v0.15.5`** (fleet floor/majority: asset template + life-architect + local ruff; corp is the lone outlier at v0.15.8), gate mode `args: []` (check-only, no `--fix`). **Deliberate bare `id: ruff` (no `name:`) for prune-safety** — matches corp so a future hub remove-leg targeting the canonical-named tombstone shape cannot silently delete this consumer gate. **Premise correction (surfaced to operator, ruled on):** the triggering prompt justified activation via "divergence-register item 9 UNRESOLVED" — recon found **no such item in ai-council's `.methodology.yaml`**, and CLAUDE.md §9 recorded the gate as **deliberately PRUNED 2026-07-04** ([#244] deploy `31e785d`). So activation **REVERSES a documented prune**; operator ruled *re-activate knowingly*. Per operator correction, the record states the item-9 pointer was **hub-side (the hub's divergence register), mis-addressed as local** — NOT "did not exist" (a canonical doc must not assert a false fact). Because the fleet-generic manifest carries **no** ruff ([#244] dropped it), a consumer ruff gate is divergent-by-definition → added a `ruff-gate` `sanctioned_divergences` entry (mirrors corp's shape). **(2) `.vscode/settings.json` carried** byte-identical from the hub (77-byte disk / sha `01b923…`, LF blob `697e12…` identical to the hub's committed blob; `.gitignore` narrowed `.vscode/` → `.vscode/*` + `!.vscode/settings.json`, mirroring the `.claude/*` un-ignore precedent). **Witnessed installed-and-firing:** probe `tests/_ruff_gate_probe.py` with an unused `import os` → `pre-commit run ruff` **Failed** (F401, exit 1); clean repo **Passed**. Probe deleted, tree clean.

**Result:** Full serial suite **444 passed** (6 deselected) via `check.ps1`; **ruff clean** (the newly-activated gate is green). `check.ps1` exits 1 at **mypy** on the **6 pre-existing** Responses-API stub-drift errors in the 3 research-provider files — **this arc has ZERO `.py` changes** (diff = 6 non-Python config/doc files), so mypy is byte-identical to `main`, **not a regression** (verified: the 3 files are unchanged vs `main`; prior 2026-07-12 entry already carries these on record). Per anti-pattern, no existing code reformatted to satisfy the new gate (recon confirmed `ruff check src/ tests/` already green). **Push HELD** per the ruling (commit-and-STOP) — local stack awaits operator review before merge.

**Changes:** `.pre-commit-config.yaml`, `.gitignore`, `.vscode/settings.json` (new), `.methodology.yaml`, `CLAUDE.md` (§9 un-prune + §12 v2.8 + `last_reviewed` 2026-07-12), `docs/intake/2026-07-06-lane-gov-functional-design.md` (row 8 amended in place, prune fact preserved), `JOURNAL.md` (this). Branch `chore/ruff-gate-reactivate-vscode-carry`, commits `5d6eb45` (ruff gate) · `2707d73` (vscode) · `61af51e` (methodology waiver) · `023b380` (docs reconcile) + this JOURNAL commit. **Not merged — push/merge held for operator review.**

---

### 2026-07-12 — Fleet ruling d1: ADR-101 R4 audit-casing gate carried (casing-only) + witnessed (merge `229f1b5`)

**Did:** Executed fleet ruling **d1** — deployed the hub `validate-hermetization` **R4 casing** gate for `docs/audits/*.md` as a consumer carry. **Verbatim carry ruled out (live finding):** the hub `scripts/validate_hermetization.py` bundles **Rule A** (top-level tree seal keyed to the HUB's `SANCTIONED_TIER1_DIRS`, which omit `src/`) — carrying it as-is would BLOCK every new `src/ai_council/` add — and **Rule B grammar** (leading date-shape + the CLOSED 11-class enum), which ai-council's free-form audit tokens (`code-quality`, `fable-*`, `council-*`, `research-*`, `fleet-recon`) don't follow. Operator ruling: carry the **R4 casing branch ONLY**. New `scripts/validate_audit_casing.py` is a faithful **extract** (casing check `^[a-z0-9.-]+$` + git glue: `git diff --cached --diff-filter=A --no-renames`, prospective-only, fail-open-loud, `--no-verify` parity); Rule A + the enum/date-grammar are dropped and **declared as machine-readable `.methodology.yaml` `sanctioned_divergences`** (`hub-hermetization-rule-a` + `hub-hermetization-rule-b-grammar`) for the hub Informant / A1 fleet_parity checker — **inert-but-recorded** (neither is a v1.3.1 consumer-manifest component; the full gate is HUB-ONLY, verified against `deploy/manifest-v1.3.1.yaml`). Enum + date-grammar **stay hub-local per d1**; a future full-gate carry needs a new operator ruling. Wired as a `repo: local` `language: system` pre-commit hook (`validate-audit-casing`, mirrors `canonical_freshness`: `always_run`/`pass_filenames:false`, computes its own staged-add set). **Witnessed installed-and-firing (real commit path, not present-on-disk):** staged `docs/audits/2026-07-12-TEST_UPPER.md` → `git commit` **REFUSED** (hook Failed, exit 1, verbatim "casing … ADR-101 R4"), HEAD unmoved; lowercase `…-test-lower.md` → hook **PASSED** (exit 0). Throwaways deleted, tree clean. Legacy quarantine `docs/audits/archive/legacy/*` (the two UPPERCASE_UNDERSCORE `_CODE_REVIEW_REPORT` files) untouched — 5 path parts, structurally outside the 3-part scope (asserted by a test).

**Result:** 18 new unit tests **green**; new files **ruff + mypy clean**; full serial suite **444 passed** (6 deselected) via `check.ps1`. `check.ps1` exits 1 at **mypy** on the **6 pre-existing BACKLOG #20** errors (Responses-API stub drift in the 3 research-provider files) — this arc has **zero `src/` `.py` changes** (diff = 2 py under `scripts/`+`tests/`, 2 yaml/config), so mypy is byte-identical to `main`, not a regression; verified by re-running mypy on `main` (same 6). Operator ruled **ship + leave #20 on record** (already filed — no duplicate added). d1 is a **partial down-payment on BACKLOG #7** (S7: reject UPPERCASE/underscore new filenames over `docs/`/`src/`) — scoped to `docs/audits/*.md` only, so **#7 stays open**. All carried pre-commit gates green on the commit (`canonical_freshness` passed; `backlog-id-on-close` passed — nothing removed).

**Changes:** `scripts/validate_audit_casing.py` (new), `tests/test_validate_audit_casing.py` (new), `.pre-commit-config.yaml`, `.methodology.yaml`, `JOURNAL.md` (this). Commit `f3f3356` on `chore/adr101-r4-audit-casing-gate`, merged `--no-ff` as `229f1b5` on `main`, pushed (`6b15c6e..229f1b5`), branch deleted. This close-out entry on `docs/journal-d1-casing-gate`, merged `--no-ff` next.

---

### 2026-07-11 — #326 consumer leg (verified no-op) + ADR-101 root-parity backfill (arc `30e4dce`..`97f5a86`)

**Did:** Two bounded methodology-conformance items in one arc. **ITEM 1 — #326 consumer leg** (operator ruling: `ARCHITECTURE.md` is CC-facing; strip ToC/Mermaid): **live re-derivation, not assumed** — a genuine end-to-end re-read (all 304 lines; canonical_freshness A2) found **no ToC block and zero Mermaid fences**; both diagrams were already converted to hand-authored compact-text under **#262 (2026-07-08)**. The only fenced block is the ASCII **Data-Flow** diagram, which is *out of the ruled scope* (ToC + Mermaid only) and **retained**. Removal is a **verified no-op**; leg closed with a re-review stamp (`30e4dce`: `last_reviewed` 2026-07-08→2026-07-11 + footer records the verdict). **Gate cascade re-derived (clean):** `toc-freshness`/`toc-generate` key on `^protocols/COUNCIL_QUESTION_GUIDE\.md$` — **no hook keys on `ARCHITECTURE.md`**; `codemap-freshness` deliberately not consumed (`.methodology.yaml` `hub-codemap-hooks` waiver); nothing removed ⇒ no roster/doc-count surface to reconcile. **ITEM 2 — ADR-101 root-parity backfill** (parity with corp-monorepo's 2026-07-11 rollout): (a) `1859d59` added the **human-visible methodology-boundary note** above §1 of `CLAUDE.md` (owner=hub = fleet methodology from `.dev-knowledge`; owner=repo = project-local; the `<!-- methodology:… owner=… -->` markers ARE the machine map; sanctioned divergences in `.methodology.yaml`) — corp carried it, ai-council did not; **marker regions untouched, body prose otherwise byte-identical**; §12 v2.7. (b) `97f5a86` committed `docs/audits/2026-07-11-technical-root-parity-disposition.md` — a 15-row table of every root file/folder vs hub + corp (CARRIED / LOCAL / IGNORE). **Table is proposals**; only IGNORE-verification (gitignore/untracked checks, no file changes) and this-session-ruled rows (#326 stamp, boundary note) were applied. IGNORE rows **verified via git**, not assumed: `.env` gitignored (`git check-ignore` exit 0) + untracked, present only as a CWD fallback for `cli.py` `load_dotenv(override=False)` (global `~/Documents/.secrets/.env` wins); caches/`.venv`/`ai_council.egg-info` all untracked; `.vscode/` absent; `assets/ruff-pre-commit.yaml` tracked (INSTALL.md §2 reference — ai-council is the origin repo). `.env` content never read or printed. **Operator amendments honored:** audit filename uses the **hub lowercase class-token** form (`YYYY-MM-DD-technical-…`), **not** corp's `_AUDIT_` uppercase form — recorded as row 15 + an open register item (corp uppercase vs hub lowercase, awaiting a fleet ruling; not propagated further); `protocols/` row flagged provisional pending **#327** (protocols-as-interface).

**Result:** Contract: pytest **426 passed** (6 deselected), `ruff` **All checks passed**. `check.ps1` exits 1 on `mypy` — the **6 pre-existing BACKLOG #20** errors (Responses-API stub drift in the 3 research-provider files); this arc has **zero `.py` source changes** (diff = 3 md files: `ARCHITECTURE.md`, `CLAUDE.md`, the new audit), so mypy is byte-identical to `main` — not a regression. All carried pre-commit gates green on each commit (`canonical_freshness` passed with both canonical docs stamped current). Zero hub/corp writes; no deletions.

**Changes:** `ARCHITECTURE.md`, `CLAUDE.md`, `docs/audits/2026-07-11-technical-root-parity-disposition.md` (new), `JOURNAL.md` (this). Commits `30e4dce`/`1859d59`/`97f5a86` on `docs/adr-101-root-parity`, merged `--no-ff` to `main` next; push after. Open register items (audit-filename convention, `protocols/` #327 genre) reported, not resolved here (core-invariant #6).

---

### 2026-07-11 — Methodology v1.3.1 rollout: first fleet consumer carried + armed + witnessed (arc `664aa91`..`288f256`)

**Did:** Rolled ai-council to methodology corpus **v1.3.1** as the Wave-1 first fleet consumer (hub ADR-101 hermetization). **Tag-ancestry pre-gate (blocking):** the intended `v1.3.0` tag (`f583509`) was verified to PREDATE the #318/#319 block-ff-push range-reconstruction fix (`00603b5`/`75a8111`/`103ce93` all NOT ancestors) — arming it would have shipped the broken pre-#318 hook; STOPPED and reported; operator cut `v1.3.1` (`84d47ab`, contains all three + hub `313500c`, on the GitHub remote), plan re-pinned. **Precommit carrier** (`664aa91`): hub hook-source `rev v1.2.0 → v1.3.1` + appended the **`block-ff-push`** pre-push gate (core-invariant #5 prevent organ, #302); `pre-commit run block-ff-push` resolved the v1.3.1 GitHub clone + passed. `codemap-freshness` from the fleet-generic install set **deliberately NOT consumed** — ai-council's codemap is hand-authored (flat single-package, no `tach.toml`) so `codemap check` always diffs; identical exclusion to corp-monorepo; `hub-codemap-hooks` is waivable. **Manual surgical carrier edit, not `deploy/tool.py`** — #276 (tool reads `.methodology.yaml` waivers) is design-deferred, so the real tool would blindly re-append the harmful hook. **`.methodology.yaml`** (`fff3619`): machine-readable codemap waiver (corp schema, read by the hub Informant). **CLAUDE.md** (`b6aaee6`): Form-A boundary markers (#312/ADR-101) grandfathered onto §1–§12, **marker-only / body prose byte-identical** (owner=hub = first-read/conventions-commit-branch/critical-rules-records/session-start-protocol; rest owner=repo); §9 reconciled to v1.3.1 + block-ff-push; §12 v2.6; `last_reviewed` re-stamped 2026-07-11. **#314** (`153b6b2`): `protocols/README.md` marks the folder project-local Council-domain (methodology protocols hub-pointer only). **#315** (`288f256`): root `INSTALL.md` resynced verbatim to the hub canonical (3 drifted sections restored; now byte-identical). **Enforcement witnessed (hard metric, isolated throwaway clone — real `main` never touched):** direct-to-main push → `block-ff-push` **REFUSED** (exit 1, "REFUSED — 1 non-merge commit(s)…"); a `--no-ff` merge push **PASSED** (gate discriminates); removing a `- [#id]` BACKLOG line uncited → `backlog-id-on-close` **BLOCKED** ("BACKLOG task(s) #1 removed but not referenced"). Throwaway removed, zero residue.
**Result:** Contract green — pytest **426 passed** (6 deselected), `ruff` **All checks passed**, all carried pre-commit gates green. `mypy` shows the **6 pre-existing BACKLOG #20** errors (Responses-API stub drift in the 3 research-provider files); this arc has **zero `.py` source changes** (diff = 5 yaml/md files), so mypy is byte-identical to `main` — not a regression. Carrier armed at **v1.3.1**. Zero hub writes.
**Changes:** `.pre-commit-config.yaml`, `.methodology.yaml` (new), `CLAUDE.md`, `protocols/README.md` (new), `INSTALL.md`, `JOURNAL.md` (this). Commits `664aa91`/`fff3619`/`b6aaee6`/`153b6b2`/`288f256` on `chore/methodology-v1.3.0-rollout`, merged `--no-ff` to `main` next. **Not pushed** — report-back first (operator go required); hub-side findings (commands `/save` gap, codemap fleet-carry gap, INSTALL.md manifest-carry, waiver-convention standardization, hub `ecosystem` version bump) reported for hub tickets.

---

### 2026-07-08 — QA lived exercise: deployed-methodology enforcement-in-effect audit (report `c194fbc`)

**Did:** Ran the first QA-class lived exercise in the repo — implemented a trivial honest feature (`clamp` util + 6 tests) on scratch branch `qa/lived-exercise-2026-07-09` and walked the full deployed loop to prove enforcement-in-effect, not presence. **Precondition PASS:** toy commit completed <2s (no WMI hang); `platform.machine()` under Python312 responsive (0.482s) — the G9 WMI stall did NOT reproduce across 6 commit-time hook invocations. **Positive path:** pre-commit + commit-msg stages fired and passed honest work (suite 432 green on the scratch branch). **Negative probes (serial, full-revert + clean-tree verify between each — parallel subagents would race the git index and breach zero-residue):** N1 floor tamper → `floor-hash-verify` BLOCKED (drift `b1bfa95 != 4d268f`); N2 remove BACKLOG `#20` with no id in msg → `backlog-id-on-close` BLOCKED at the commit-msg stage (pre-commit passed first); N3a same-day ARCHITECTURE body edit → PASSED (A2 date-granular, same-day-safe by design), N3b genuine stale stamp → `canonical_freshness` A2 BLOCKED (teeth confirmed); N4 direct commit on `main` → **NOTHING BLOCKED** (landed `6a44191`, hard-reset away, never pushed) = FINDING F1; N5 `/codex-review` resolved+ran end-to-end (codex 0.141.0/gpt-5.5, clean on the toy diff) + `/review-closures` resolved+ran (surfaced 1 WEAK #10, approved nothing). **Action 3 degraded leg:** SessionStart:resume `pre_commit install` fails in-venv (`No module named pre_commit`) — the G8 venv gap (`[dev]` declared in `pyproject.toml` but not reinstalled into this venv), NOT WMI; the floor-guard leg is exit 0; the commit-time mesh is intact via the Python312 shim.
**Result:** Verdict — the deployed methodology enforces in effect; every commit-time gate probed BLOCKED its target violation with HEAD unmoved + a verbatim refusal. Exceptions: F1 (no consumer-side commit-time branch protection on `main` — routed to hub), F2 (resume-arm G8 venv gap — self-heals on next `pip install -e ".[dev]"`; commit-time unaffected), F3 (pre-push stage armed-but-empty), plus a by-design `canonical_freshness` same-day granularity nuance. WMI (G9) currently dormant. INSTALL.md provenance (Action 0): **RUNBOOK ARTIFACT** (deploying commit `73e9a48`; referenced by `.claude/settings.json` + the ruff asset) — not an orphan, no deletion word needed. The report seeds the audit-class taxonomy question → routed to the hub hermetization ADR. Zero residue: all injections reverted, scratch branch (local + `origin`) deleted, `main`=`origin/main`=`fba7b13` untouched by the probes.
**Changes:** `docs/audits/2026-07-09-qa-lived-exercise.md` (new, `c194fbc`), `JOURNAL.md` (this). Commit on `docs/qa-lived-exercise`, merged `--no-ff` to `main` next. Scratch `qa/lived-exercise-2026-07-09` (toy feature `0b77a67`, probe commits `bd3a17c`/`6a44191` — ephemeral, branch deleted) NOT merged.

---

### 2026-07-08 — SessionStart pre_commit arm: `[dev]` declaration + WMI CLI-hang diagnosis (G8/G9, commit `d6dc783`)

**Did:** SessionStart:resume hook `python -m pre_commit install` failed non-blocking (`.venv python: No module named pre_commit`). Diagnosed (G8): the hook AND the layer-6 verify both use bare `python`, context-dependent — Python312 (has `pre_commit`) in the original session, the `.venv` (lacked it) under an active venv on resume; the git-hook shim's hardcoded `INSTALL_PYTHON` (the only deterministic path) was exercised by neither. Fix: declared `pre-commit>=4.5` in `pyproject.toml [dev]` (durable, hub-floor-matched; chosen over the machine-specific INSTALL_PYTHON hook-repoint, which would fork corp-monorepo's bare-python wiring). Verifying via the hook's own command line surfaced a deeper **MACHINE-level** hang (G9): `pre_commit/languages/golang.py:42` → `platform.machine()` → Python 3.12 `platform._wmi_query()` stalls on this box's WMI (faulthandler-confirmed; identical on the Python312 path; intermittent — 6 ai-council commits + hub merges landed today via the armed commit-time hooks). Runtime verify marked **BLOCKED-by-WMI** with a verbatim retry command; the offline `.venv` copy used for diagnosis was reverted (SessionStart back to a fast non-blocking fail). Appended G8+G9 to the Wave-1 gap-notes; routed the WMI stall to the hub as a machine-scoped ENVIRONMENT gotcha.
**Result:** Commit-time pre-commit hooks fired normally through this arc (`canonical_freshness`/`backlog-id-on-close` passed). The `[dev]` declaration makes `pre_commit` resolvable in-venv on a fresh `pip install -e ".[dev]"`. #215 attestation scope clarified: commit-time mesh FIRED; SessionStart:resume auto-arm is the WMI-degraded leg.
**Changes:** `pyproject.toml`, `docs/intake/2026-07-08-runbook-gap-notes.md` (G8 update + G9). Commit `d6dc783` on `fix/sessionstart-precommit-venv`, merged `--no-ff` to `main` next.

---

### 2026-07-08 — Wave-1 onboarding pilot (n=1): 6-layer verify + conformance convergence (#281/#282/#262) + #215 attestation (arc `7769025`..`c0265b6`)

**Did:** Executed `docs/runbooks/repo-onboarding.md` (post-A-S3) end-to-end as the runbook's first real test — #131 pilot + #215 conformance, kept split. **Layers 1–6 verified green** (already deployed v1.2.0 2026-07-04): floor guard `--require-present` exit 0, `.gitignore` negations intact, no root-floor orphan; carriers present; tier1-lifecycle plugin enabled + `/review-closures` resolves; review-profile recorded (CLAUDE.md §8); REGISTRY row + `deployed-versions.yaml` present (confirm-only, **no hub write**); all 3 hook stages armed + `import pre_commit` OK. **#282** (`7769025`): `.gitattributes` (`* text=auto eol=lf`, `*.ps1 eol=crlf`), `--renormalize` no-churn. **#281** (`2511329`): BACKLOG Track-X → ADR-66 named themes (zero structural `Track` strings), stable `[S<n>]` ids S1–S9 — ai-council is the **first consumer to adopt the #286 S-id grammar** (ADR-99 clause A); re-filed **#110 + #128** MOVED from the hub (ADR-41, hub `ea6217a`); hub S-id validator `OK (6 themes, 9 stories, 18 tasks, 0 warn)` via a **scratch copy** (Q2 — no in-repo validator hand-copied; carrier gap G1). **#262** (`d5c4e25`): both ARCHITECTURE Mermaid blocks → **HAND-AUTHORED** compact-text (16 nodes / 14 edges / 5 layers preserved), LF — the codemap CLI degenerates to a 2-orphan stub on ai-council's flat no-`tach.toml` layout (gap G4; generator-managed NOT met, honest scope). **Taxonomy** (`b849653`): `docs/intake/` created (operator GO), 6 census-named intake artifacts relocated from `docs/audits/` (2 audit-flavored kept); **no `docs/handoffs/`** (ADR-60/42; census target → hub NEEDS-RULING). Orphan `.claude/worktrees/fable-audit/` **SURFACED, not deleted** (gated on operator word). **#215 attestation** (pinned hub `main` @ `9e6ceb6`, read-only, gitignored logs only): `enforcement_coverage --fire --run-date 2026-07-08` **FIRED** (session_end_backpressure + canonical_freshness enforcing-local; floor/plugin/global-config wired; doc_claims/git_backlog_drift hub-scoped, not faked); `floor_conformance` **9/9 PASS** (poisoned+deleted floor caught, hook auto-arms, real task branch→commit→gate→merge); `fleet_health` OK. `observe-arc` n-of-6 env-constrained (billed child, timed out 4m40s) and `audit.py repo` report non-persist — both filed as gaps. **Gap-notes** (`c0265b6`): 7 gaps + honest-scope closures → operator for hub filing.
**Result:** `check.ps1`: pytest **426 passed** (6 deselected), `ruff` **All checks passed**; `mypy` shows the **6 pre-existing BACKLOG #20** errors (Responses-API stub drift in the 3 research-provider files) — this arc has **zero `.py` source changes**, so mypy is byte-identical to `main` (not a regression). 11 files changed, all docs/config.
**Changes:** `.gitattributes`, `BACKLOG.md`, `ARCHITECTURE.md`, `docs/intake/` (6 relocations + README + gap-notes), `JOURNAL.md` (this). Commits `7769025`/`2511329`/`d5c4e25`/`b849653`/`c0265b6` on `chore/wave-1-onboarding`, merged `--no-ff` to `main` (HEAD = merge). Push + branch delete after. Orphan deletion + #281/#262/G-series hub disposition await the operator.

---

### 2026-07-07 — Consumer-local hook-type arming: commit-msg + pre-push now armed (hub #275, merge `2add141` anchored)

**Did:** Closed the SessionStart arming-gap flagged as an unresolved hub-side candidate in the two prior entries (`71e1307`, `dc411c7`): `pre-commit install` locally only placed the `pre-commit` hook, so the hub-sourced `backlog-id-on-close` (a **commit-msg**-stage hook) stayed wired-but-dormant. Fix executed consumer-side per ADR-41: added `default_install_hook_types: [pre-commit, commit-msg, pre-push]` to `.pre-commit-config.yaml` top level (4-line addition, YAML-only — zero Python touched). Ran `pre-commit install` (global Python312 `pre-commit` 4.5.1 — note the repo `.venv` does **not** carry `pre_commit`; `py`/`python`/`.venv` all lack the module, the SessionStart hook and manual installs run the Python312 exe at `~/AppData/Local/Programs/Python/Python312/Scripts/pre-commit.exe`). Verified all three `.git/hooks/{pre-commit,commit-msg,pre-push}` now exist with the framework signature. **Trip test (real block, HEAD-anchored):** on the feature branch, scratch-removed the `[#20]` `BACKLOG.md` task line, staged, and committed with a message carrying **no** `[#id]` — `backlog-id-on-close` **FAILED at the commit-msg stage** (exit 1, "commit-msg: BACKLOG task(s) #20 removed but not referenced in the message"), commit **blocked**, HEAD unmoved at `5d6c54c`. Reverted cleanly (`git restore`) — `#20` restored, tree clean.
**Result:** Trip evidence confirms the previously-dormant commit-msg stage now fires. pytest **426 passed** (6 deselected, unchanged), ruff clean. `check.ps1`'s mypy leg shows the 7 pre-existing errors of **BACKLOG #20** (OpenAI-SDK Responses-API type-stub drift in the 4 research provider files + the `types-PyYAML` stub) — unrelated to this YAML-only change; mypy state is identical to `main` (branch diff vs `main` = the 4-line YAML addition only).
**Changes:** `.pre-commit-config.yaml`. Commit `5d6c54c` on `chore/precommit-arm-hook-types`, merged `--no-ff` as `2add141` on `main`, this close-out commit next. Push next; branch delete after.

---

### 2026-07-06 — Arc 3 item 6 adjudicated + executed: pre-commit hub-hook source pinned by URL+rev (merge `71e1307` anchored)

**Did:** Operator adjudicated the item-6 fork (recommendation delivered same-day, see prior entry): pin-by-URL+rev. Changed `.pre-commit-config.yaml`'s hub-hooks `repo:` from the relative sibling path `../.dev-knowledge` to `https://github.com/rdwornik/dev-knowledge` (rev unchanged, `v1.2.0`); the hub's own v1.2.0 manifest already declared this URL as the precommit carrier's target and the origin is real/reachable — ai-council was the outlier. Verified **enforcement-in-effect, not presence**: `pre-commit clean` (full cache wipe) + `pre-commit install --install-hooks` confirmed a fresh clone from the URL (not the old path); `pre-commit run --all-files` passed clean; a deliberate stale-TOC edit to `protocols/COUNCIL_QUESTION_GUIDE.md` made `toc-freshness` FAIL (exit 1, correct diff) from the URL-sourced hook, then reverted (`git checkout --`, confirmed clean). `backlog-id-on-close` proved both directions in an isolated scratch repo (never touched real `BACKLOG.md`): FAILS removing a `[#id]` task with no message reference, PASSES with `closes [#id]` present — scratch dir removed after.
**Result:** pytest **426 passed** (unchanged), ruff clean, gate green. The SessionStart arming-gap (pre-commit stage only installed locally, so `backlog-id-on-close`'s commit-msg stage stays dormant until the enforcement-mesh carrier arms all 3 hook types) is unrelated to this transport change and remains a hub-side candidate (ADR-41 — not touched here).
**Changes:** `.pre-commit-config.yaml`. Commit `a8f7ac5` on `chore/precommit-url-pin`, merged `--no-ff` as `71e1307` on `main`, this close-out commit next. Pushed.

---

### 2026-07-06 — Arc 3 conformance residuals: Track rename + hygiene sweep (merge `dc411c7` anchored)

**Did:** Executed the Arc 3 dedicated-chat scope (hub ADR-99 Track-X convention + hub BACKLOG L234 residual pointer block + the 2026-07-06 L-GOV measurement audit), items 1-5, on `chore/track-rename-conformance-residuals`. (1) Renamed "Epic X" -> "Track X" in the two LIVE files that used it (`BACKLOG.md` -- the ADR-66 story-map's schema-level term throughout; `ARCHITECTURE.md` -- one "Last updated" reference); JOURNAL, the 5 dated `docs/audits/*.md` files, and `docs/decisions/ADR-10/11/12.md` excluded by construction (historical/immutable). (2) Removed 6 stale `.claude/settings.local.json` permission entries hardcoding the pre-move `/c/.../Documents/Scripts/ai-council` path (repo now at `Documents/Dev/ai-council`); generic entries kept. (3) `worktrees/fix/` confirmed already absent -- no-op. (4) Closed the v1.2.0 manifest gap: `backlog-id-on-close` was missing from the `../.dev-knowledge` hub-hooks pull (only `toc-freshness`/`toc-generate` were wired) -- added it; config verified via `pre-commit run --all-files`. Codemap hooks deliberately left unwired: hub JOURNAL (2026-06-03) already ruled ai-council's hand-authored codemap is the correct end-state (BACKLOG #79, resolved) -- wiring the generator hook would regress that ruling, not fix a gap. Recorded as a hub-side candidate (not fixed here): the SessionStart hook only arms the default `pre-commit` stage (`python -m pre_commit install`), so `backlog-id-on-close` (commit-msg stage) is wired-but-dormant locally until the enforcement-mesh carrier's install command arms all 3 hook types the way the hub's own `scripts/arm_hooks.py` already does fleet-side. (5) A2-stamp check: no file was actually in A2 FAIL; the only issue was `CONTRIBUTING.md` in A1 WARN (34 days stale). Genuine re-reads of `ARCHITECTURE.md` (tripped A2 by the item-1 edit) and `CONTRIBUTING.md` both found the same real staleness -- their "Pre-commit setup"/"Validators" sections named only `normalize-headers`, omitting `floor-hash-verify`, `canonical_freshness`, and the hub-sourced TOC/backlog hooks already live in `.pre-commit-config.yaml`. Fixed both, bumped both stamps to today, and updated `CLAUDE.md` SS9 to match (v2.5, re-stamped) since it had the identical gap.
**Result:** Gate clean -- `canonical_freshness` 0 fails, 0 warns (was 1 WARN pre-arc). pytest **426 passed** (6 deselected, unchanged), ruff clean. `grep -ri "epic "` across live files returns zero misses. Item 6 (relative-path vs URL+rev pre-commit source fork) delivered separately as a recommendation-only doc per the arc's frozen scope -- not executed.
**Changes:** `BACKLOG.md`, `ARCHITECTURE.md`, `.claude/settings.local.json`, `.pre-commit-config.yaml`, `CLAUDE.md`, `CONTRIBUTING.md`. Commits `b33faee`/`3bc75a4`/`34522ab`/`0732493` on `chore/track-rename-conformance-residuals`, merged `--no-ff` as `dc411c7` on `main`, this close-out commit next. Not yet pushed.

---

### 2026-07-06 — Technical refactoring guide persisted to docs/audits (`docs/audits/2026-07-06-code-refactoring-guide.md`)

**Did:** On operator request (the code-quality audit read too verdict-level/functional — wanted the code-level "how"): authored the technical companion on `docs/code-refactoring-guide` (`dad4a31`). Twelve concrete refactorings — **A1-A5 structural** + **B1-B7 mechanical** — each with exact `file:line`, faithful before/after code re-read from live `main` (`anthropic`/`openai`/`grok_research`/`openai_mini_research` re-read to ground the snippets), steps, and a testable done-when. Document only — nothing applied; each `R#` is its own future branch session.
**Result:** Docs-only; pytest unit suite **426 passed** (unchanged), pre-commit green incl. canonical_freshness; tree clean. Part A unblocks Wave-C: A1 provider template base (→ CliProvider #16, ADR-gated), A2 `cli.main` decomposition (→ D2 parity + doctor; the `--file` gap falls out), A3 unify the two error classifiers + wire the dead policy trio + kill the `_config.timeout_sec` reach-through, A4 `save_to_file` split (→ verdict package), A5 break the `runner`↔`orchestrator` re-export. Part B = fast hygiene: research-helper hoist, `utcnow`×5, naive `datetime`×7, dead code, mypy type-args, W1 flake isolation (the real-`~/Downloads` scan), `RunPolicy`-from-config.
**Changes:** `docs/audits/2026-07-06-code-refactoring-guide.md` (new), `JOURNAL.md` (this). Commit `dad4a31`, merge `55f9b8c` + this close-out commit on `main`. Pushed. Zero source/config/test changes.

---

### 2026-07-06 — Code-quality audit of `src/` persisted to docs/audits (`docs/audits/2026-07-06-code-quality-audit.md`)

**Did:** Ran the read-only implementation-quality audit of `src/ai_council/` on `docs/code-quality-audit` (`e1f178a`), companion to the same-day architect intake. Phase-1 mechanical evidence (radon cc/mi, vulture cross-checked against real callers, mypy --strict, an ast function-length pass, the internal import graph, and consistency/config/efficiency censuses) + Phase-2 first-hand reads of all five load-bearing surfaces + two scoped subagents (provider-family duplication; test-suite quality). Installed radon+vulture into `.venv` (analysis-only). Report = grade **B+ (engineered, not patched — duplication is the debt, not accretion)**, five refactor seeds each with a testable done-when, and a §6 not-findings list referencing the known items without refiling.
**Result:** Docs-only; pytest unit suite **426 passed** (6 deselected; unchanged count), pre-commit green incl. canonical_freshness; tree clean. Load-bearing verdicts: `cli.py` REFACTOR-FIRST (277-LOC F(72) `main`, two drifted request paths — the D2 `--file` gap is a symptom of that split); `providers/` WORKABLE+seed (~85% boilerplate, no template method, `xai`≡`deepseek`); `output.py` WORKABLE (clean `_write_routed`/`_save_metrics_json` seams; `save_to_file` E(33) the watch); `healthcheck.py` SOLID (MI 79.51; doctor reuses as-is); `orchestrator`/`runner` WORKABLE (`runner`→`orchestrator` re-export soft cycle). Flake W1 root-caused (verified first-hand): non-hermetic — scans the real `~/Downloads` (`scan_downloads:true` not disabled, `load_config` uncached), lives at `test_research.py:1632` — upgrades W1 from "ordering flake" to test-isolation defect.
**Changes:** `docs/audits/2026-07-06-code-quality-audit.md` (new), `JOURNAL.md` (this). Commit `e1f178a`, merge `5213d04` + this close-out commit on `main`. Pushed. Zero source/config/test changes.

---

### 2026-07-06 — Technical-architect intake document assembled (`docs/audits/2026-07-06-technical-architect-intake.md`)

**Did:** Assembled the architect entry-point document on `docs/architect-intake` (`1ca5466`): embedded the functional architect's content block verbatim, verified every Reading-Map path against live `main` (FR-master marked `[pending: never committed]`; one historical-audits row added), corrected the Topic Router against real lane-doc headings (9 pointer refinements, `seats[]` → §3(Q5) the substantive one), added Evidence: lines to all 5 FR clusters and all 16 draft ADRs, and verified register statuses live — corrected "ADR-01..10 Proposed" to the true spread (only 09/10 Proposed), flagged the hub-carrier R1–R8 count and the CLI-4 label collision. Housekeeping: stale `docs/adr-11-12-ratification` branch deleted (`-d`, merged).
**Result:** Docs-only; pytest 426 passed (unchanged count), pre-commit gates green incl. canonical_freshness; tree clean. Merged `--no-ff` as `3dea2d4`, branch deleted, pushed at close-out.
**Changes:** `docs/audits/2026-07-06-technical-architect-intake.md` (new), `JOURNAL.md` (this). Commit `1ca5466`, merge `3dea2d4` + this close-out commit on `main`. Pushed.

---

### 2026-07-06 — Five parallel lane-design worktrees integrated to main (`--no-ff` × 5)

**Did:** Integration pass in the primary checkout for the five parallel lane functional-design worktree sessions. Verified state (five branches, each exactly one commit on `5c81e71` adding one disjoint `docs/audits/2026-07-06-lane-*-functional-design.md`; tree clean), then merged each serially `--no-ff`, no conflicts: L-EPI `d4a5fb7`→`81f60ec`, L-CLI `9b8bfa4`→`5bdd439`, L-DOC `617483a`→`b45084f`, L-INT `22ae4d6`→`5ccb9b0`, L-GOV `5576cae`→`e4485e1`. This entry retroactively anchors the five logged ADR-85 session-end overrides from those worktree sessions. Torn down all five worktrees + branches (`worktree remove` / `prune` / `branch -d`).
**Result:** Docs-only session — five new lane-design docs, zero source/config change; pytest + pre-commit green with unchanged counts; tree clean; only the primary worktree remains and no `worktree-lane-*` branches survive. Pushed `origin/main` at close-out (L-GOV standing-rule candidate: push at close-out merges — also clears the stale-worktree-base hazard that hit three of the five sessions today).
**Changes:** `docs/audits/2026-07-06-lane-{epi,cli,doc,int,gov}-functional-design.md` (new), `JOURNAL.md` (this). Five `--no-ff` merge commits `81f60ec`/`5bdd439`/`b45084f`/`5ccb9b0`/`e4485e1` + this close-out commit on `main` (direct-to-main per the integration-pass operator instruction, per fleet-recon precedent). Pushed.

---

### 2026-07-05 — ADR-11 + ADR-12 ratified; invocation contract authored; Wave-0 doc reconciliation (D4)

**Did:** Docs-only execution session on `docs/adr-11-12-ratification` (operator ratification authority granted in the session prompt; one checkpoint, approved with 4 adjustments). C1 `165b94a`: GUIDE Wave-0 reconciliation — all 4 example frontmatters aligned to the true default (dropped `models:` + `synthesizer: openai`; detection keys retained per inbox-sniffing convention), single labelled explicit-override example added, mechanism text corrected (synthesizer is EVICTED from the panel, not swapped), explicit "effective default = 4 debaters + gemini non-participating synthesizer" statement added. C2 `68d0571`: ADR-11 (delegated invocation contract) ratified — Accepted (2026-07-05), Related carries the fleet-recon reconciliation line (D1–D3 HOLD). C3 `9938161`: `protocols/COUNCIL_INVOCATION_CONTRACT.md` authored — lanes A/B, flag set verified against live cli.py, frontmatter precedence, exit codes 0/1/2/3 with caller obligations, artifacts, JSON payload, degradation/RoutingError semantics, hub-WHEN/WHY vs protocols-HOW boundary, Lane-A caller walkthrough, MANDATORY Known-deviations section naming both D2 parity gaps (`--file` frontmatter leak; research `--return-dir` no-op). C4 `c11fb42`: ADR-12 ratified with the fleet-recon §7 markup applied verbatim (v1 adapters = claude+codex; four witnessed safety invariants incl. scratch-cwd-primary-isolation and identity-or-no-seat; gradient codex > claude > grok(post-OAuth) > agy(excluded); per-call pin rule; §5 default-flip stays evidence-gated; gemini seat path struck). C5 `3325a8a`: README index +2 Accepted rows; VISION References line unbound (ADR-01 onward → index pointer, cannot re-stale at ADR-13). C5b `10dd355`: canonical_freshness A2 gate correctly blocked the next commit (C5 edited VISION without re-stamping `last_reviewed`) — genuine end-to-end VISION re-review performed, `last_reviewed` → 2026-07-05, and the review caught real staleness: the research-provider list still named the deprecated o4-mini/o3 deep-research models (migrated 2026-05-18; fixed to gpt-5.4-mini/gpt-5.5). VISION line 25 dual-output framing flagged for the next currency pass, not changed.
**Result:** Zero source-code changes (`git diff --stat main` = protocols/ + docs/decisions/ + VISION.md + JOURNAL.md only); pytest unit suite + ruff green — 426 passed (first full run flaked once on `test_inbox_exits_3_when_any_batch_run_degraded`, which passes in isolation and on the full re-run; ordering-dependent, unrelated to a docs diff — watch for recurrence); all pre-commit gates green at merge time (TOC freshness on the GUIDE edit; canonical_freshness after the C5b re-review — see C5b for the one blocked commit, working as designed). ADR-13 untouched (still a draft inside the 07-04 audit only). BACKLOG untouched — precedent verified (ADR-09's BACKLOG edit closed #12; ratifying ADR-12 closes nothing, #16 remains open until a CLI backend runs a debate turn). Status divergence recorded for a future hygiene pass: ADR-09/10 still say Proposed while 11/12 say Accepted. Deferred staleness named: CLAUDE.md §11 still lists local ADRs only through ADR-08 (its own currency pass; edits there trip the canonical-freshness gate).
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
