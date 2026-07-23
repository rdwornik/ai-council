# Pre-handoff repo audit + cleanup — night batch

**Date:** 2026-07-22 (batch ran into 2026-07-23; filename per commissioning prompt)
**Branch:** `chore/pre-handoff-cleanup` off `main` @ `8888d9e` · **Landing:** commit-and-STOP, not merged
**Lanes:** A doc-currency (luna) · B backlog/journal/git (luna) · C archival (luna) · D dead code (terra) · E test suite (terra + first-hand measurement) · F handoff readiness (first-hand)
**Discipline:** every scanner finding re-verified first-hand before any fix; luna produced 3 false positives (see Checker requirements #12).

---

## 1. FIXED

- `31e08c7` **Lane E** — `test_timeout_counts_as_failure` genuinely waited 30.07s (21% of suite wall time): it patched `hc._DEFAULT_TIMEOUT_SEC`, but `_ping_timeout` prefers `provider._config.timeout_sec` (MockProvider sets 30s) capped at `_MAX_TIMEOUT_SEC`, so the patch was dead. Both knobs now patched; assertions unchanged. File: 30.07s → 0.16s.
- `13525b2` **Lane A** — ARCHITECTURE.md reconciled to reality: codemap/responsibilities/layer tables gain `boost` + `crux_check` (both shipped modules were absent from every structural table); `config` path corrected to repo-root `config/` (was `src/ai_council/config/`, which does not exist); CLI surface `run + boost + doctor`; pre-commit roster completed to the live 12 hook ids (+`ruff`, +`block-ff-push`); research table model names reconciled to source/settings (`o4-mini-deep-research`→`gpt-5.4-mini`, `o3-deep-research`→`gpt-5.5`, grok →`grok-4.20-0309-reasoning`); invariant 1 reworded (models.py holds dataclasses + `CruxStatus` enum + `CruxChecker` protocol — "pure dataclasses only" was false); invariant 5 now documents the `pick_synthesizer()` `is_participant=True` last-resort fallback (`runner.py:69-75`) that the same doc's Key Design Decisions already recorded; local ADR span `01…08`→`01…14`. `last_reviewed` re-stamped after genuine end-to-end re-read with source verification.
- `f9d6985` **Lane A** — VISION.md boost→decide paragraph now states the honest limitation as shipped: gaps become advisory `[BOOST-GAP]` annotations for the panel; the boost never enumerates options or invents constraints the caller did not supply (verified against `boost.py:102-116`). CLAUDE.md §11 gains the missing **ADR-13** roster line (Accepted 2026-07-18; ADR-12/ADR-14 were rostered, ADR-13 never was) + §12 v2.13 history entry; both stamps re-stamped after genuine re-reads (§10 code anchors re-verified live: `merger.py:201`, `settings.yaml:324`, `debate.py:59`).
- `6eddb49` **Lane A** — `protocols/COUNCIL_QUESTION_GUIDE.md` dangling pointer repaired: "See `README.md` Transcript Routing" → ARCHITECTURE.md § Transcript Routing (README deleted per ADR-38 A5, `505ea94`). #86/#87 deliberately untouched (filed).

Lanes B, C, D: zero autonomous fixes by design (report-only / approval-gated).

---

## 2. PROPOSED (by leverage)

1. **Panel-default authority conflict** · code default is `default_panel` = 3 models (`settings.yaml:13`, `runner.py determine_panel`), `--full` is a live flag selecting `full_panel` (`cli.py:486`) — while ADR-02, ARCHITECTURE:~189 ("Full 5-model panel is the default; `--full` is a no-op") and GUIDE (~74-85, 475) all claim full-5 default · **Rec:** resolve under #4 (already fired per 2026-07-21 grooming): amend ADR-02 (Revised) or restore config, then fix ARCHITECTURE + GUIDE in one pass · **Cost if wrong:** callers budget for a 5-model spend profile they don't get, or debates silently run under-panelled.
2. **pytest-xdist: declare + document** · installed (3.8.0) but NOT in `pyproject.toml` dev extras; measured 139.94s → 35.76s (3.9×), 818 passed under `-n auto` (no order-dependence surfaced) · **Rec:** declare in `[dev]`, document `-n auto` in `testing.md`/`check.ps1` · **Cost if wrong:** occasional xdist worker-crash flakiness; ~5s overhead on tiny runs. (Rail 3: no dep added by this batch.)
3. **ADR-11 Deployment-Status stamp** · stamp (2026-07-18 inventory) still lists #35 as open remainder; #35 was struck 2026-07-19; file substantively amended 2026-07-22 · **Rec:** one dated amendment-marker line refreshing the stamp · **Cost if wrong:** none — additive marker.
4. **`2026-07-17-epi1-archaeology-KEY-SEALED.json` placement** · root-level JSON violates the audits README three-class invariant (date-slug **markdown** / `archive/` / registered corpus) · **Rec:** relocate under the registered `2026-07-17-epi1-archaeology/` corpus at the prescribed unseal, not before (sealed instrument; #67-guarded) · **Cost if wrong:** breaks the seal workflow — operator-only.
5. **`scripts/council-ask.ps1`** · referenced by nothing (rg across src/tests/scripts/config/docs/hooks/packaging/JOURNAL/BACKLOG: zero hits) and hardcodes `C:\Users\1028120\...` · **Rec:** delete · **Cost if wrong:** an undocumented human/out-of-repo caller breaks.
6. **Production-unused symbols** · `research/cache.py:133 cache_invalidate` (tests-only) and `model_string()` (`providers/base.py:291`, `research/provider.py:36` + impls; used only by tests + the two manual `verify_openai_*` scripts) · **Rec:** keep-or-remove ruling; if kept, mark as deliberate library surface · **Cost if wrong:** an external importer breaks.
7. **`feat/vscode-boundary-colors` branch** · fully merged (`merge-base --is-ancestor` exit 0; `main..branch` empty) · **Rec:** delete local branch · known pre-existing, operator-pending · **Cost if wrong:** nothing — commits stay reachable from main.
8. **Genuine copy-paste duplication** (next code session, not tonight): `_format_duration()` (`research/display.py:25` ≡ `research/output.py:28`) · `_collect_annotations()` ×3 research providers (`grok_research.py:148`, `openai_deep_research.py:151`, `openai_mini_research.py:150`) · `LegResult`/`_ok`/`_fail` ×3 verifier scripts · xai/deepseek `_invoke` byte-identical but decree-protected (CLAUDE §5.10) — excluded.
9. **Research-provider hardcoded model defaults vs invariant 3 / VISION Values** · `openai_deep_research.py:34` (`"gpt-5.5"`), `openai_mini_research.py:34` (`"gpt-5.4-mini"`) are constructor fallbacks; invariant says "none hard-coded" · **Rec:** either strip the fallback defaults (code) or scope invariant 3 to runtime-selected values · judgement, not done tonight.
10. **Invariant 2 (`cli.py` performs no business logic)** · `cli.run()` is a 196-statement complexity-50 function doing provider loading, health checks, mode resolution, research dispatch (= live P1-12) · **Rec:** either reword the invariant honestly or file the P1-12 refactor; don't leave a false invariant standing.
11. **Test-suite structural options** (measured, not impressionistic): the four e2e git-subprocess families (`validate_audit_casing` ~27s, `validate_docs_registry` ~28s, `validate_sealed_keys` ~18s, research-CLI exit-code ~10s) dominate the post-fix profile · **Rec (either/both):** an `e2e` marker for a fast dev loop; session-scoped template-repo fixture cloned per test (needs coupling review) · **Cost if wrong:** cross-test state coupling — why it is not an autonomous fix.
12. **Collection time** · 7.81s, driven by `import ai_council.cli` = 3.37s (top-level SDK imports) · **Rec:** lazy provider imports if the dev loop matters · code change, propose only.
13. **8 live-unfiled P1s** from the 2026-07-20 night code audit (see §Lane B below): P1-6, P1-10, P1-11, P1-12, P1-13, P1-14, P1-15, P1-16 · filing is the next window's scoped decision.
14. **Unreachable git arcs** · `git fsck --unreachable --no-reflogs`: 83 commits / 45 tip arcs (WIP/probe/spike history incl. pre-rescue material) · **Rec:** no prune without explicit operator confirmation; list preserved in Lane D output.
15. **Empty `.claude/worktrees/`** · zero entries; watched by `.vscode` excludes · **Rec:** leave (harmless, auto-recreated) or remove — one-line call.

Report-only (no action proposable by this batch): JOURNAL SHA notes (below), historical stale paths inside immutable audits/ADRs (`2026-05-17` audit → old guide path; `ADR-09:30` → old rubric path — amendment-marker candidates at most), intake `runbook-gap-notes` refs to two absent files, CLAUDE.md §12 v2.10 citing the gov1 register at its pre-archival path (true when written; file now at `docs/intake/archive/`).

### Lane B detail (report-only lane)

- **Drifted-closed open items: 0.** Reservation note present (`BACKLOG.md:157`); #84/#85 free; #86/#87 assigned as renumber successors. PASS.
- **#82 premise re-checked first-hand: TRUE** (not falsified) — `metrics.py:70` itself carries the `NOTE (#82)` separate-ledger comment.
- **7 open items re-confirmed still-live against source:** #69 (`cli.py:688,812-815`), #75 (`output.py:268-272`), #76 (`output.py:1264-1268`), #79 (`orchestrator.py:288-294`), #80 (`output.py:1024-1026`), #81 (`output.py:1008-1045`), #70 (`research/output.py:58-60`).
- **P1 accounting (prompt said "9 never filed"; true numbers differ):** the audit holds **16** P1s. 6 fixed — deliberately, with in-code `P1-n:` comments (P1-1, P1-2, P1-3, P1-7, P1-8, P1-9; spot-verified `debate.py:311-317`, `orchestrator.py:153-180`), 2 were pre-filed (P1-4=#69, P1-5=#75), **8 live-unfiled** (P1-6 degraded-cache, P1-10 summarizer no-timeout, P1-11 timeout substring-sniff, P1-12 `cli.run()` complexity, P1-13 `output.py` four-modules, P1-14 tautological test, P1-15 blind-voting test asserts nothing, P1-16 orchestration mocked out of existence).
- **JOURNAL SHA citations:** luna listed 15 "unresolvable" SHAs. Sample adjudication: `2b21cb26`, `00603b5` = **hub-repo citations** (cross-repo, correct as written); `feedbac` = **regex false positive** (word fragment); `4864b70` = the already-documented repaired decoration; `9be4c35` (JOURNAL:1797, `chore/witnessed-opmin-verify` tip) = **unadjudicated — the one candidate genuine blemish**; remaining 10 unadjudicated (JOURNAL is append-only — any repair is a dated correction entry, next window). 31 recent merge SHAs are not named verbatim in JOURNAL — JOURNAL names the underlying session SHAs instead; not judged a defect.

---

## 3. CHECKER REQUIREMENTS (spec for the claim-vs-reality checker)

Each rule below caught (or would have caught) a real drift found tonight:

1. **Module-table completeness** — every `src/ai_council/*.py` module appears in ARCHITECTURE's codemap + responsibilities + layer tables. *(caught: boost, crux_check absent from all three)*
2. **Path existence** — every backtick-quoted repo path in canonical docs (CLAUDE, ARCHITECTURE, VISION, protocols/) resolves on disk; allowlist for hub-qualified paths (`.dev-knowledge/...`) and historical sections (§12, JOURNAL, dated audits). *(caught: `src/ai_council/config/`, GUIDE→README.md, gov1-register move)*
3. **Hook-roster parity** — set-equality between hook ids in `.pre-commit-config.yaml` and the rosters in ARCHITECTURE §Validators and CLAUDE §9. *(caught: 10 vs 12 — `ruff`, `block-ff-push` missing)*
4. **ADR-roster parity** — `docs/decisions/ADR-*.md` on disk == CLAUDE §11 roster == ARCHITECTURE Governing-ADRs line == decisions README index. *(caught: ADR-13 unrostered)*
5. **Config-claim parity** — doc statements of defaults (panel, synthesizer, model strings) anchored to `settings.yaml` keys; flag doc≠config. *(caught: 5-vs-3 panel conflict, o3/o4 stale model names, grok variant)*
6. **CLI-surface parity** — `@main.command` registrations in `cli.py` == subcommands named in docs. *(caught: `boost` absent)*
7. **Stamp honesty** — `last_reviewed` / `Deployment-Status` date ≥ file's last substantive commit; ADR deployment stamps are outside the current `canonical_freshness` A2 scope — extend to them. *(caught: ADR-11 stamp pre-dating its own amendment + listing struck #35)*
8. **SHA-citation resolution** — `[0-9a-f]{7,}` word-bounded tokens in JOURNAL/BACKLOG must resolve in-repo OR sit in a hub-qualified sentence. Hex-only + word-boundary kills the `feedbac` class of false positive. *(caught: the 15-item list, and its own FP)*
9. **Durations regression gate** — flag any unit test exceeding a threshold (e.g. 10s). The dead-patch 30s test would have been flagged the day it regressed. *(caught: test_timeout_counts_as_failure)*
10. **Dep parity** — venv-installed vs pyproject-declared. *(caught: pytest-xdist installed, undeclared)*
11. **Invariant spot-checks as assertions** — each ARCHITECTURE invariant gets a grep-able verify line (e.g. invariant 1: no `def ` with body logic in models.py beyond dataclass/enum/protocol). *(caught: invariants 1, 5 false as written)*
12. **Scanner-verification discipline (process rule)** — every automated finding carries a re-runnable evidence command; tonight luna produced 3 false positives (`check_floor_hash.py` "missing" — it exists at `.claude/check_floor_hash.py`; CLAUDE:239 "unqualified path" — it is hub-qualified; `feedbac` as SHA). A checker that reports without evidence commands recreates the audits-never-absorbed failure mode.

---

## 4. MEASUREMENTS

Before any change (serial, `-m "not integration and not envcheck"`):
- Collected: 825 (7.81s collection; `import ai_council.cli` alone = 3.37s)
- Suite: 818 passed, 1 xfailed, 6 deselected in **139.94s**
- Profile: top-25 tests ≈ 116s of 140s. #1 `test_timeout_counts_as_failure` **30.07s**; e2e git-subprocess validator families ≈ 71s; ~800 remaining tests ≈ 24s.
- `-n auto` (xdist 3.8.0, pre-fix): **35.76s**, 818 passed.

After (healthcheck fix): `test_healthcheck.py` 9 passed in 0.16s (was >30s). Full-suite landing number in §6.

Known-correct deselections confirmed: 5 envcheck + 1 billed integration; markers declared in `pyproject.toml` and applied in `test_api_keys.py` / `test_integration.py`; no unmarked network/provider test found; remaining `asyncio.sleep(5|10)` occurrences are cancelled hang-simulators that never elapse (durations floor of top-25 = 0.24s).

---

## 5. DECISIONS THIS PROMPT DID NOT COVER

1. **Base drift:** prompt stated base `main @ 0f1a150`; main was at `8888d9e` (the colors-journal merge landed after the prompt was written). Branched from current `8888d9e`.
2. **Date rollover:** batch crossed midnight into 2026-07-23. Report keeps the prescribed `2026-07-22-*` filename; `last_reviewed` stamps use the true review date 2026-07-23.
3. **Codex lane mapping:** luna/terra resolved to `codex exec --sandbox read-only -c model=gpt-5.6-luna|-terra` (the convention witnessed in `~/.claude/bin/codex-review.ps1`); sol not used, per prompt. All verdicts read from output-file bodies, never console counters.
4. **CLAUDE.md/VISION.md re-stamps:** editing canonical docs forces `last_reviewed` bumps through `canonical_freshness`; both re-stamps were backed by genuine end-to-end re-reads with live-state verification (recorded in CLAUDE §12 v2.13), not blind refreshes.
5. **"9 P1" premise corrected** rather than force-fitted: 16 total / 6 deliberately fixed / 2 filed / 8 live-unfiled.
6. **GUIDE 5-panel claims left unfixed** although textually fixable — entangled with the #4 authority question; fixing text before the authority ruling would just pick a side silently.
7. **ADR-07 archival not proposed as urgent:** its header already carries `Status: Superseded by ADR-43` + stamp; Lane C's "archive" is cosmetic relocation only.
8. **Orphan-reference audit files (7) left in place:** they are admissible class-(a) date-slug records; "referenced by nothing" alone is not an invariant violation.

---

## 6. LANDING VERIFICATION

Close-out (2026-07-23, after final commit):
- Validators: `validate_audit_casing.py` exit **0** · `validate_sealed_keys.py` exit **0** · `validate_docs_registry.py` exit **0** · `validate_backlog.py BACKLOG.md` exit **0** (OK: 7 themes, 13 stories, 49 tasks, 0 warnings).
- Full unit suite: **825 collected — 818 passed, 1 xfailed, 6 deselected** (floor met exactly) in **59.36s** serial.
- **Measurement caveat, recorded for honesty:** the 139.94s "before" number in §4 was taken while four codex scan processes ran concurrently (CPU contention), so the raw before/after delta overstates the healthcheck fix. The uncontaminated attributable saving is the test's own measured call duration: 30.07s → ~0.05s. Uncontended serial baseline going forward: **59.36s**; xdist `-n auto` reference point: 35.76s (itself measured under the same contention, so likely also pessimistic).
- Working tree clean; branch NOT merged, NOT pushed.

---

## AMENDMENT (2026-07-23, session-close archaeology) — proposal §2.1 was WRONG; rule §3.5 refined

**§2.1 ("Panel-default authority conflict") is withdrawn as a false finding.** The Block-2 archaeology traced the full chain: `default_panel` has been the 3-model set since its introduction (`b513cad`, 2026-02-21, originally `["claude","gemini","deepseek"]`) and ADR-02's Revised (2026-05-11) **Implementation section explicitly documents the indirection** — "The 5-model effective default is achieved via `cli.py`: when `--lite` is not passed, `use_full_panel or not lite` evaluates to `True`, selecting `full_panel` … `default_panel` in config remains the 3-model lite set." Live code confirms: `cli.py:689` and `cli.py:812` compute `eff_full = (use_full_panel or not lite) or …`, so a bare invocation resolves to **`full_panel` (5-model)** on both the `--file` and inbox paths. Code, ADR-02, ARCHITECTURE ~189, and the GUIDE's 5-panel claims are **all mutually consistent**; this report (and the Lane A scan it verified) read `determine_panel()` + `settings.yaml` in isolation and never traced the flag. `--full` is a no-op except in the `--lite --full` combination, where it wins — consistent with "kept for backward compat."

**Consequently §3 rule 5 (config-claim parity) is refined:** the checker must adjudicate doc claims against **effective flag-resolution behaviour**, never raw config key values alone — a raw-key comparison would mechanically reproduce this exact false positive. This is the batch's **fourth** scanner-class false positive, strengthening rule 12.

**What genuinely remains in this area:** #4's ADR-02 amendment (overlap policy; its condition fired 2026-07-21) and ADR-02's `No open remainder` stamp — already filed. The confusing name `default_panel` (it is the *lite* set) is a naming nit, not a defect. _(Amendment is additive; §2.1 above is preserved unedited as the record of what the batch believed.)_
