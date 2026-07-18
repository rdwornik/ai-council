---
version: "1.0"
owner: rob
last_reviewed: "2026-07-19"
status: active
---

# Night-consolidation verification — the 2026-07-18 shipped batch

> **Purpose.** Prove *empirically* — with the command run and the verbatim output — what the
> ~20 merges of 2026-07-18 actually deliver at runtime, convert the proof into a re-runnable
> checker, clear the review debt, and refresh doc currency. **Merged is not deployed;** every
> "shipped" claim below rests on witnessed behaviour, not on a commit message or a green gate.
>
> **Immutability (CLAUDE.md §5.3):** *"ADRs, transcripts, handoffs, and audits are immutable —
> supersede with a new file or an in-file amendment marker; never edit in place."* This audit is
> therefore a **new** dated file; nothing under `docs/audits/` was edited. **Session-end gate
> (this lane):** `scripts/session_end_backpressure.py` HARD-blocks the Stop unless the `JOURNAL.md`
> entry for the arc names a session commit-SHA (only `/override` exits).
>
> **Exclusion zone honoured:** `docs/audits/2026-07-18-cli4-parity/**` and every `SEALED-KEY*.json`
> were never read, written, or staged. The operator's blind #27 scoring is untouched.
>
> **No fixes.** Every finding became a `BACKLOG.md` item (#59–#68). The only code added is the
> checker (`scripts/verify_night_consolidation.py`). No new folders, no new ADRs/intakes, `main`
> untouched.

## TL;DR

- **All 8 verification legs PASS** at $0, each carrying the exact command + verbatim decisive output.
- Legs were proven by **live execution of the shipped code** (direct calls / `MockProvider` / canned
  CLI outputs), never by reading diffs. Paid multi-provider debates and the gated `backend: cli`
  flip were deliberately avoided — the flip is the ADR-12 §5 default-flip, gated on the pending #27
  scoring, and is out of this lane's authority.
- An independent Codex derivation (`sol`, blind to everything but `src/`) **agrees with all 8 PASS
  verdicts** and adds adversarial edge-case gaps. A Codex review (`terra`, #44 review debt)
  confirms three of them and adds one. **No divergence** where a leg marked PASS cannot hold.
- **Terra is live** (probe returned `TERRA-OK`): the "credits exhausted until 2026-07-23" premise is
  empirically false, so #44 was executed now rather than left date-gated.
- The checker reproduces all 8 verdicts on its own run (**8/8 PASS, exit 0, idempotent**).

## Method and cost posture

The eight surfaces under test are about parsing, output routing, sidecar contents, directory
resolution, and refactors — none depends on whether a model gives a *good* answer. So each leg was
proven by exercising the **real shipped function** with controlled inputs and inspecting the **real
emitted artifact**:

- **$0, offline** — no provider was called for a debate. `MockProvider` (`tests/conftest.py:101`,
  `token_count=10`, `AsyncMock`-shadowed) and canned CLI JSON/stderr drive the real pipeline.
- **CLI-seat lane gated.** The committed `config/settings.yaml` routes *every* seat to `backend=api`
  (no `backend: cli` / `cli_command`). Activating CLI seats needs the §5 flip, which is gated on the
  #27 parity scoring — so the end-to-end seat run is UNVERIFIED-by-design and recorded as a GAP, and
  #41 is witnessed at the adapter level instead (the honest offline proof).
- **Fallbacks watched.** No CLI-seat fallback occurred in any run (a STOP condition); the only live
  API contact was one `council doctor` health ping during L4 (a diagnostic, not a debate), whose
  stray `./output/` was deleted and removal verified — working tree clean.

## What shipped on 2026-07-18 (the batch under test)

| Item | Change | Merge |
|---|---|---|
| #22 | `--file` routes through `parse_file` — YAML frontmatter stripped, precedence CLI>frontmatter>config | `3e64e81` |
| #23 | research honors `--return-dir` (dispatch collapsed to one `_run_research_dispatch`) | `04cf534` |
| #39 | `--no-persist` + `AICOUNCIL_OUTPUT_DIR` output guard + `output/health/` keep-10 retention | `e140c5d` |
| #40 | verdict-package `options_considered` sourced for pick + ideas | `edce0e7` |
| #41 | real token counts recorded for the codex + claude CLI seats | `edce0e7` |
| #42 | research slug strips a leading "research" token (no doubled prefix) | `5b139ad` |
| #45–#48 | re-export shim broken · `utcnow`→`iso_now()` · dead `_target_projects` deleted · RunPolicy from `settings.yaml` | `02b6dd0` |
| ADR-13 | verdict package stamps `contract_version = "1.0"` (§7 empty) | `5dd4782` |
| #20 | openai 2.x mypy **stopgap** (real fix still OPEN) | `3e1005f` |
| — | synthesizer default gemini→openai (operator-ratified) | `6e83e41` |

## Wave 2 — verification legs (each PASS carries a command + verbatim output)

**L1 — #22 `--file` frontmatter strip + precedence** · VERDICT **PASS**
`parse_file` on a crafted `---\nsynthesizer: grok\nrounds: 3\nmode: judge\n---\nWhich datastore…`:
```
TEXT_REPR: 'Which datastore should we pick?'   (no --- , no synthesizer: line -> no leak)
META: {'synthesizer': 'grok', 'rounds': 3, 'mode': 'judge'}
CONFIG_DEFAULT_SYNTHESIZER: 'openai'
CLI_WINS  (flag=openai, meta=grok) -> 'openai'
FRONTMATTER_WINS (flag=None, meta=grok) -> 'grok'
CONFIG_WINS (flag=None, meta={}) -> 'openai'
```

**L2 — #23 research `--return-dir` (both dirs, canonical first, identical)** · VERDICT **PASS**
`save_research_to_file(report, canon, from_cache=False, return_dir=ret)`:
```
CANONICAL_MD5: f517b9c3c63a03d0540c9b1366d5feae LEN=441
RETURN_MD5:    f517b9c3c63a03d0540c9b1366d5feae LEN=441
CONTENTS_MATCH: True   (canonical written first per _write_routed output.py:161-183)
```

**L3 — ADR-13 `contract_version == "1.0"`** · VERDICT **PASS**
`save_to_file` then `save_verdict_package` → open the emitted JSON:
```
VERDICT_FILE: council-verdict-20260719_...-pick-should-we-use-yaml-or-json-for-config.json
contract_version = '1.0'
```

**L4 — #39 output guard + health retention keep-10** · VERDICT **PASS** (with a filed finding)
`_prune_health_records` over 12 `doctor-<ts>.json` + `doctor-latest.json`:
```
_HEALTH_RETENTION = 10 ; before=13 -> after=10 timestamped + doctor-latest.json kept
oldest_remaining=doctor-20260703_120000.json (2 oldest pruned)
```
Finding (→ #65): the `run` command honours `AICOUNCIL_OUTPUT_DIR`/`--no-persist`/`--output`
(`cli.py:517-529`), but the **`doctor` command ignores them** — it calls `run_doctor` without
`output_dir`, so health records always land in canonical `./output/health/`, contradicting
`doctor.py`'s own module docstring. `--no-persist` on a `run` is UNVERIFIED (paid path) → #66.

**L5 — #40 `options_considered` on pick AND ideas** · VERDICT **PASS**
`_build_verdict_payload` on a pick result (question has `## Options`, synthesis has none) and an
ideas result (synthesis `## Top Tier`):
```
PICK  options_considered.items = ['Redis', 'Memcached', 'Postgres']   (question fallback)
IDEAS options_considered.items = ['Idea Alpha', 'Idea Beta']          (synthesis Top Tier)
```

**L6 — #41 non-zero token counts, both CLI seats (adapter level)** · VERDICT **PASS** (seat lane gated)
Canned `.modelUsage`/`.usage` into `ClaudeCliProvider._extract`; canned stderr into `CodexCliProvider._extract`:
```
CLAUDE input=4600 (100+4000+500) output=200 token_count=4800
CODEX  token_count=1234 output_tokens=1234 (parsed "tokens used\n1,234")
```
GAP: the CLI-subscription seat lane is inactive in the committed config; the end-to-end seat run is
gated on the §5 flip (tracked by #27). This leg proves the parse logic, not production seat routing.

**L7 — #42 research filename, no doubled prefix** · VERDICT **PASS**
`save_research_to_file` with `query="research best vector databases in 2026"`:
```
council-out-20260719_...-research-best-vector-databases-in-2026.md   (research- x1, no research-research)
```

**L8 — #45/#46/#47/#48 code-quality residue** · VERDICT **PASS** (all four sub-checks)
```
#45 orchestrator.CouncilRunner OK ; from ai_council.runner import CouncilRunner -> ImportError ; runner keeps build_all_providers/determine_panel
#46 iso_now() tz-aware (+00:00) ; 5 call sites (gemini/grok/openai_deep/openai_mini/perplexity) ; 0 live datetime.utcnow(
#47 routing.py: no _target_projects ; only self._known set from the constructor
#48 from_config({5,3})->5/3 ; from_config(None)->2/1 ; from_config(load_config().policy)->2/1 (matches settings.yaml:19-20)
```

## Mechanism — `scripts/verify_night_consolidation.py`

Wave 2 is codified as a deterministic, offline, re-runnable checker following the `scripts/`
verify/validate sibling convention (`validate_backlog.py`, `validate_audit_casing.py`,
`verify_openai_deep.py`). It prepends the co-located `src/` to `sys.path` (so it always tests *this*
repo's code, not a sibling worktree's editable install), exercises each shipped path, and prints a
PASS/FAIL table. Run: `py scripts/verify_night_consolidation.py`. Its own run reproduces Wave 2:

```
LEG  ID       VERDICT  EVIDENCE
L1   #22      PASS     no_leak=True; tiers cli=openai/fm=grok/cfg=openai
L2   #23      PASS     paths=2 canonical_first=True identical=True
L3   ADR-13   PASS     contract_version='1.0'
L4   #39      PASS     keep=10 before=13 after_ts=10 latest_kept=True
L5   #40      PASS     pick=[Redis,Memcached,Postgres] ideas=[Idea Alpha,Idea Beta]
L6   #41      PASS     claude token_count=4800; codex token_count=1234
L7   #42      PASS     ...-research-best-vector-databases-in-2026.md (research- x1)
L8   #45-48   PASS     shim-ok iso_now-ok deadcode-ok RunPolicy-ok
RESULT: 8/8 PASS   (exit 0; second run also exit 0, tree stays clean)
```

## Wave 3 — CC ↔ sol adjudication (divergence by divergence)

Codex `sol` derived guarantees + gaps for all 8 surfaces from `src/`+`config/` **only** (blind to
this audit, the BACKLOG, and Wave 2). It **agrees with every PASS verdict**; it adds adversarial
"does-not-guarantee" gaps. There is **no leg where sol says a PASS cannot hold** — so nothing is
escalated; the added gaps are filed.

| Leg | CC (Wave 2) | sol (Wave 3) | Verdict | Adjudication |
|---|---|---|---|---|
| L1 #22 | PASS | agrees; +malformed FM uncaught, invalid mode→hardcoded `pick` | agree | **merge** → #64 |
| L2 #23 | PASS | agrees; +return-dir write best-effort/swallowed | agree | **merge** → #62 (also terra T4) |
| L3 ADR-13 | PASS | agrees (hardcoded "1.0"); +no shared version constant | agree | **adopt-CC**; +terra T5 → #63 |
| L4 #39 | PASS +doctor-gap | agrees; **same doctor gap** + non-strict prune bound | agree | **merge** → #65 |
| L5 #40 | PASS | agrees; +empty `## Options` heading blocks fallback | agree | **merge** → #60 (also terra T1) |
| L6 #41 | PASS (gated) | agrees; +`None`→0, codex all-as-output, all seats api | agree | **merge**; §5 flip → #27, `None`→0 → #61 |
| L7 #42 | PASS | agrees; **+bare "research" query still doubles** | agree | **adopt-sol** → #59 |
| L8 #45-48 | PASS | agrees; +iso_now scope, defaults-if-absent | agree | **adopt-CC**; terra CLEAN confirms |

## Wave 3b — terra (#44 review debt)

**Probe:** `codex exec -c model=gpt-5.6-terra "Reply with exactly: TERRA-OK"` → `TERRA-OK` (exit 0,
"tokens used" shown). The "credits exhausted until 2026-07-23" premise is **empirically false** —
terra is live and $0-billable under the subscription, matching the 24 zero-fallback CLI debates.
So #44 was **run now** (not left date-gated). Read-only, findings only:

- **T1 (#40, HIGH, output.py:610):** a non-empty synthesis `## Options` with no top-level bullets
  suppresses the question fallback → `options_considered.items = []`. → #60 (confirms sol).
- **T2 (#41, HIGH, cli_base.py:283):** partial Claude `usage` (e.g. `input_tokens: null`) → definitive
  `token_count=0`; `metrics.py` books zero. → #61 (confirms sol).
- **T3 (#45–#48):** **CLEAN** — confirms L8.
- **T4 (#23, HIGH, research/output.py:102 / output.py:178):** research `--return-dir` is best-effort
  and never verified, unlike the verdict package's R4 fail-loud; a read-only return dir is swallowed
  and `run_research` still succeeds. → #62 (triple-confirmed: CC happy-path + sol + terra).
- **T5 (contract_version, HIGH, output.py:387):** a metrics-sidecar write failure in
  `_save_metrics_json` aborts `save_to_file` before the caller reaches `save_verdict_package`, so no
  `contract_version: "1.0"` package is emitted at all. → #63.

**#44 disposition:** the review ran; **#45–#48 are CLEAN**; the other surfaces are functionally correct
on the happy path (Wave 2) with the edge-case hazards above filed. The date-gated #44/#33 waivers may
now be re-dispositioned by the operator against this live terra pass.

## Verified empirically vs merely merged

| Claim | Status | Evidence |
|---|---|---|
| #22 frontmatter strip + 3-tier precedence | **VERIFIED** | L1 verbatim |
| #23 research `--return-dir` both dirs | **VERIFIED (happy path)** | L2 verbatim; failure path best-effort (#62) |
| ADR-13 `contract_version="1.0"` | **VERIFIED** | L3 verbatim; emission-ordering hazard (#63) |
| #39 retention keep-10 | **VERIFIED** | L4 verbatim |
| #39 `AICOUNCIL_OUTPUT_DIR`/`--no-persist` on `run` | **MERGED, not live-witnessed** | code-verified (`cli.py:517-529`); UNVERIFIED live → #66; `doctor` ignores it → #65 |
| #40 `options_considered` pick+ideas | **VERIFIED** | L5 verbatim; empty-heading edge (#60) |
| #41 CLI-seat token counts | **VERIFIED (adapter)** / **MERGED (end-to-end)** | L6 verbatim; seat lane gated on §5 flip (#27); `None`→0 (#61) |
| #42 no doubled prefix | **VERIFIED** | L7 verbatim; bare-"research" edge (#59) |
| #45–#48 residue | **VERIFIED** | L8 verbatim; terra CLEAN |
| synthesizer gemini→openai default | **VERIFIED** | `load_config().defaults.synthesizer == 'openai'` (L1) |

## Findings filed to BACKLOG (next free local id was #59)

All findings are edge-cases/robustness gaps or verification debt — **there were zero leg FAILs**.
Filed as open tasks (no code fixed):

| id | story | finding | source |
|---|---|---|---|
| #59 | [S16] | research slug: a bare "research" query still yields `research-research` (#42 regex needs separator+content) | sol |
| #60 | [S16] | non-empty `## Options` synthesis heading with no bullets suppresses the question fallback → `[]` (#40) | sol+terra |
| #61 | [S16] | partial CLI `usage` (`input_tokens: null`) booked as `token_count=0` (#41) | sol+terra |
| #62 | [S10] | research `--return-dir` write best-effort/swallowed — no R4 fail-loud parity (#23; extends #35 to research) | CC+sol+terra |
| #63 | [S10] | metrics-sidecar write failure aborts before verdict-package emission (ADR-13/#26) | terra |
| #64 | [S10] | malformed frontmatter uncaught + invalid `--file` mode → hardcoded `pick`, not configured default (#22) | sol |
| #65 | [S11] | `council doctor` ignores `--output`/`--no-persist`/`AICOUNCIL_OUTPUT_DIR` vs its docstring (#39) | CC+sol |
| #66 | [S16] | witness `--no-persist`/`AICOUNCIL_OUTPUT_DIR` live on a `run` (offline-only today) | verification debt |
| #67 | [S7] | pre-commit guard rejecting a staged `SEALED-KEY*.json` (proposal) | this session |
| #68 | [S7] | pre-commit guard blocking a new folder / new `docs/` path without an operator-authorization marker (proposal) | this session |

## Proposed guards (PROPOSED, not created — taxonomy from primary sources)

Neither is built; both are filed (#67, #68). Their taxonomy home is derived and quoted:

1. **Reject a staged `SEALED-KEY*.json`.** The 2026-07-18 consolidation grooming log records the real
   near-miss: *"sealed-key leak caught + untracked before merge"* (`BACKLOG.md`, S14/smoke-pair entry).
   Home = the repo's pre-commit surface + a `scripts/` validator, exactly as CLAUDE.md §9 lists the
   existing consumer-local gates: *"`validate-audit-casing` (consumer-local … `scripts/validate_audit_casing.py`, `always_run`)"*.
   A `scripts/validate_no_sealed_key.py` staged-file gate mirrors that pattern.
2. **Block a new folder / new `docs/` path without an authorization marker.** CLAUDE.md §5 item 9:
   *"No leftovers — any automated or scratch-creating process … removes and verifies removal of
   everything it created before it counts as done."* and the anti-pattern §10: *"Running validators
   with no args — vacuous pass; always pass `--all` or specific paths."* Home = the same pre-commit +
   `scripts/` surface; `docs/` taxonomy is fixed by **ADR-60** (`decisions/` + `audits/` + `archive/`,
   README-seeded), so a new top-level `docs/` child is presumptively a taxonomy violation absent a marker.

## Wave 1 — evidence sweep (reconciled with verification)

- **Doc currency (1A → verified):** `VISION.md:24` and `CLAUDE.md:203` still named **Gemini** as the
  synthesizer — genuinely stale (default is `openai`, ratified 2026-07-18). **Both fixed this session**
  with `last_reviewed` re-stamps. `ARCHITECTURE.md:175/196` "DRAFT-INT-1/#26" was flagged stale but is
  **NOT** — DRAFT-INT-1 is the verdict-package design; ADR-13 ratified DRAFT-INT-2 (`contract_version`),
  a different artifact. Left unchanged. `CONTRIBUTING.md`: clean.
- **Deployment-Status stamps (1B → reconciled):** the "open: #20" stamps on the two 2026-07-06 audits
  were flagged stale but are **CORRECT** — #20 is a *stopgap-applied, real-fix-open* item
  (`BACKLOG.md:60`; grooming log: *"#20 stays open"*). No stamp is genuinely stale; the night-batch
  and lane/ADR stamps are all current. **No stamp edits needed.**
- **Link integrity (1C):** **0 broken references from today's churn.** The `docs/smoke/` → `docs/audits/`
  re-home left zero dangling refs. The 14 flagged items are pre-existing absolute `C:/…/file.py:NN`
  code-citations inside **immutable archived** audits (`docs/audits/archive/*`) — not links, not from
  today; non-actionable (and the files are immutable).
- **BACKLOG hygiene (1D → reconciled):** next free **local** id is **#59** (grooming log: "after #57 →
  #58"; #110/#128 are re-filed *hub* ids, not the local sequence — the "#129" estimate was wrong).
  #20 is correctly open (not closed-but-listed). Structure parses as the ADR-66 story-map.

## Decisions in force (and what each forbids)

- **ADR-12 §5 (CLI-backend default flip is evidence-gated).** Forbids flipping any seat to
  `backend: cli` as the default until the #27 parity scoring ratifies it — so this lane may **not**
  activate CLI seats, and #41's end-to-end proof stays deferred.
- **ADR-13 (invocation-contract versioning).** The verdict package must stamp `contract_version`
  ("1.0" while CONTRACT §7 is empty). Forbids emitting a verdict package without the version field /
  silently changing the shape without bumping it.
- **ADR-10 / DRAFT-INT-1 R4 (deterministic return + fail-loud).** Canonical `./output/` is always
  written first and the hub is never a silent default; a *required* `--return-dir` miss on the verdict
  package must raise, not exit 0. Forbids treating the canonical write as optional or a required return
  as best-effort (the research path's gap here is #62).
- **CLAUDE.md §5.3 (immutability) + §5 item 9 (no leftovers).** Forbids editing an ADR/transcript/
  handoff/audit in place, and forbids any process leaving scratch/folders behind.

## What must happen next (and why)

1. **Operator: score #27 and decide the §5 flip** — it unblocks the only end-to-end-UNVERIFIED claim
   (#41 CLI-seat tokens in production) and the whole CLI-cost lane. This audit de-risks it (adapter
   parse proven; terra live; zero fallbacks).
2. **Re-disposition #44/#33** against this live terra pass (the credits-exhausted premise is false).
3. **Groom #59–#66** — small robustness edges; #62 (research return-dir fail-loud) and #63
   (metrics-failure blocks verdict) are the highest-value (silent-failure hazards).
4. **Decide #67/#68** — the SEALED-KEY and new-folder guards; the near-leak was real.

## LESSONS captured

Appended to `LESSONS.md` (append-only): worktree isolation is triggered by *side effects*, not task
size; a trial's parameter space must be enumerated as one matrix or the gates arrive serially; an
order that could create a path must name the path; `git add -A` nearly leaked a sealed artifact;
verification prose belongs in a script (this audit's checker); and — this session — a "live witness"
that would cost money or need a gated flip is honestly served by exercising the shipped code with
`MockProvider`/canned inputs at $0, with the un-exercisable remainder recorded as an explicit GAP.

## Comprehension probe (Wave 5)

A cold reader — Codex `sol`, given **only** this document in an otherwise-empty directory (no chat
context, no other summaries) — answered the operator's four questions from the report alone.
**Result: 4/4 ANSWERED — no PARTIAL, no UNANSWERABLE. The report is self-contained; no gap to close,
no re-probe needed.**

| # | Question | Verdict |
|---|---|---|
| Q1 | What shipped 2026-07-18 + evidence per item | ANSWERED |
| Q2 | Verified empirically vs merely merged | ANSWERED |
| Q3 | What must happen next, and why | ANSWERED |
| Q4 | Three decisions in force + what each forbids | ANSWERED |

Raw cold-reader answers (verbatim, logged as evidence):

> **Q1:** "#22 file/frontmatter precedence—L1 output; #23 return-dir—L2 matching MD5s; #39 retention—L4
> kept 10, while output guards are only code-verified; #40 options—L5 pick/ideas outputs; #41 token
> counts—L6 Claude=4800/Codex=1234 at adapter level; #42 slug—L7 'research- x1'; #45–48
> cleanup/refactors—L8 four checks; ADR-13—L3 emitted `contract_version='1.0'`; synthesizer OpenAI
> default—L1 config output. #20's stopgap is evidenced only by merge `3e1005f` …" — **ANSWERED**
>
> **Q2:** "Empirically verified: #22, #23 happy path, ADR-13 happy path, #39 retention, #40, #41 adapter
> parsing, #42, #45–48, and the OpenAI synthesizer default. Merely merged/not live-witnessed: #39 run
> output guards and #41 end-to-end CLI-seat routing. The report warns: 'Merged is not deployed.'" —
> **ANSWERED**
>
> **Q3:** "Score #27 and decide the §5 CLI flip to unblock #41 end-to-end proof; re-disposition #44/#33
> because Terra is live; groom #59–#66, prioritizing silent-failure hazards #62/#63; decide #67/#68
> because the SEALED-KEY near-leak was real." — **ANSWERED**
>
> **Q4:** "ADR-12 §5 forbids defaulting seats to CLI before #27 ratification. ADR-13 forbids unversioned
> verdict packages or silent shape changes without a version bump. ADR-10/DRAFT-INT-1 R4 forbids
> optional canonical writes and best-effort handling of a required return directory." — **ANSWERED**

The cold reader's Q2 split independently reproduces this audit's verified-vs-merged table — including
that #39 run-path guards and #41 end-to-end are *merged, not live-witnessed* — confirming the
distinction survives a context-free read.
