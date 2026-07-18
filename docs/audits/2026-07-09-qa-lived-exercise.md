---
kind: qa-audit
generated: 2026-07-08
labeled: 2026-07-09
head_commit_at_start: fba7b13bbb2d45420ace9ec1efa35d1071ae6006
status: complete
owner: Rob
---

# QA lived exercise — does the deployed methodology actually bite?

> **Deployment-Status (2026-07-18 inventory):** DEPLOYED — F1 block-ff-push live (v1.3.1); gates witnessed blocking; residual: F2/F3 self-heal untracked. _(Additive inventory stamp; body below unchanged.)_

> **First QA-class audit in this repo.** Its existence seeds the audit-class
> taxonomy question (is `qa-audit` a distinct `docs/audits/` kind vs the existing
> code/architecture/methodology audits?) — **routed to the hub hermetization ADR**;
> not adjudicated here.
>
> **Date note (honest):** executed **2026-07-08** (system clock + git author date);
> the exercise contract labeled it **2026-07-09**. The branch
> (`qa/lived-exercise-2026-07-09`) and this filename keep the contract's `07-09`
> identifiers for findability; every dated fact below is the true `07-08`.

## Intent

Prove **enforcement-in-effect** for the full deployed loop by living it: implement a
trivial honest feature on a scratch branch and verify every deployed gate/hook/skill
both **passes honest work** and **blocks violations**. Presence is never the verdict;
a gate never seen to block is unproven (enforcement_coverage doctrine, generalized).

## Method & honesty notes

- **Serial, not parallel, negative probes.** The contract suggested one subagent per
  probe row. Five subagents committing on one working tree would race the git index
  and corrupt state — and violate the zero-residue boundary. Git-mutating probes were
  run **serially in the main thread**, with a full revert + clean-tree verification
  between each. Serial execution is also *more* valid: no cross-probe contamination
  (this mattered — see N3, whose landed commit would otherwise have armed a
  `canonical_freshness` block on a later probe). Subagent-class depth went into the
  read-only mechanism mapping instead.
- **Every commit attempt ran under a 90 s timeout** so a WMI hang could not trap the
  branch. None hung.
- **Report merges via its own `docs/` branch → `--no-ff`** (Action 5). The scratch
  branch and its toy feature are deleted whole; nothing from it reaches `main`.
- **Verbatim capture artifact:** the Windows console renders the gate scripts' em-dash
  (`—`) as `�` under cp1252 (the known `CLAUDE.md` §10 trap). Outputs below show the
  **true** character; the `�` was a capture-layer cosmetic only.

## Precondition — WMI / pre-commit completes on a scratch commit

**PASS.** The toy-feature commit (`0b77a67`) completed in <2 s with all hook stages
firing; the branch push completed; a direct probe of the G9 root cause
(`platform.machine()` under Python 3.12) returned `AMD64` in **0.482 s**. The WMI hang
(G9) **did not reproduce** in any of the 6 commit-time hook invocations this session.
Exercise cleared to proceed.

## Action 0 — `install`-file provenance

**File:** `INSTALL.md` (repo root, tracked). **Verdict: RUNBOOK ARTIFACT — not an
orphan, not a deploy-manifest carrier.**

| Evidence | Finding |
|---|---|
| Deploying commit | `73e9a48` (2026-06-02) `chore: install Tier-1 lifecycle plugin + ruff gate asset (ADR-70 #73 Unit-5a/5b)` — the sole commit in its `--follow` history |
| Content | The install runbook for the `tier1-lifecycle` plugin + the ruff-gate asset (marketplace add, plugin install, `settings.json` shape, `pre-commit install`, verify steps) |
| Live references (tracked) | `.claude/settings.json` (its `//` note points here for the ruff gate), `assets/ruff-pre-commit.yaml` (the asset it instructs merging), `docs/audits/2026-07-02-methodology-adoption-audit.md` |
| Describes live components | The `tier1-lifecycle@dev-knowledge-methodology` plugin it installs **is enabled** in `.claude/settings.json` |

Not a manifest carrier (contains zero manifest lines — it is human-facing instructions).
Not an orphan (referenced by live config + describes an installed, enabled component).
**→ No operator deletion word required.**

## Probe matrix

| Probe | Expected | Observed | Verdict |
|---|---|---|---|
| **Precondition** WMI/scratch commit | completes, no hang | commit `0b77a67` <2 s; push clean; `platform.machine()` 0.482 s | **PASS** |
| **A1** positive path (toy feature) | all stages fire + pass honest work | pre-commit 3 checks (2 skip-by-scope, `canonical_freshness` **Passed**) + commit-msg `backlog-id` **Passed**; suite **432 passed** | **PASS** |
| **N1** tamper 1 floor byte → commit | `floor-hash-verify` blocks | **Failed** exit 1, HEAD unmoved, drift `b1bfa95… != 4d268f…` | **BLOCKED-as-designed** |
| **N2** remove BACKLOG `#20`, no id in msg → commit | `backlog-id-on-close` trips at commit-msg | pre-commit passed **first**, then commit-msg **Failed** exit 1, HEAD unmoved | **BLOCKED-as-designed** |
| **N3a** edit ARCHITECTURE body, no stamp bump → commit | (freshness behavior) | **Passed**, landed `bd3a17c` — A2 is date-granular & same-day-safe | **PASS (by design)** |
| **N3b** genuine stale stamp (`last_reviewed`→07-01) → commit | teeth: A2 blocks | `canonical_freshness` **Failed** exit 1, HEAD unmoved | **BLOCKED-as-designed** |
| **N4** direct commit on `main` (trivial file) | record consumer-side behavior | **nothing blocked** — commit `6a44191` **landed** on `main` | **FINDING (F1)** |
| **N5a** `/codex-review` | resolves + runs | codex v0.141.0 / gpt-5.5 / read-only, **exit 0**, dated audit artifact; toy diff → all bands `(none)` | **PASS** |
| **N5b** `/review-closures` | resolves + runs | skill loaded; gate `surface` **exit 0**, surfaced 1 WEAK `#10`; **approved nothing** | **PASS** |
| **Action 3** SessionStart:resume arm | degraded (per G8/G9) | leg-1 floor guard exit 0; leg-2 `pre_commit install` (venv) **exit 1** `No module named pre_commit`; WMI dormant | **DEGRADED (G8, not WMI) — F2** |

## Verbatim gate outputs

**A1 — positive path (commit `0b77a67`):**
```
Normalize dated-log entry headers....................................(no files to check)Skipped
Verify CLAUDE-FLOOR.md matches its sha256 sidecar....................(no files to check)Skipped
canonical_freshness last_reviewed gate (A2 FAIL blocks the commit).......................Passed
TOC freshness (markdown doc vs its own headers)......................(no files to check)Skipped
Require [#id] in commit message when a BACKLOG task is closed.......................Passed
[qa/lived-exercise-2026-07-09 0b77a67] feat(qa): add clamp pure-function util + tests ...
 2 files changed, 60 insertions(+)
```

**N1 — floor tamper (BLOCKED; HEAD unmoved):**
```
Verify CLAUDE-FLOOR.md matches its sha256 sidecar........................................Failed
- hook id: floor-hash-verify
- exit code: 1

floor hash drift: b1bfa95b78f1 != sidecar 4d268f329a7e -- regenerate via the hub generator,
or restore with `git checkout HEAD -- .claude/CLAUDE-FLOOR.md`.
```

**N2 — backlog task removed, no id in message (BLOCKED at commit-msg; HEAD unmoved):**
```
canonical_freshness last_reviewed gate (A2 FAIL blocks the commit).......................Passed
...
Require [#id] in commit message when a BACKLOG task is closed.......................Failed
- hook id: backlog-id-on-close
- exit code: 1

commit-msg: BACKLOG task(s) #20 removed but not referenced in the message.
  Add [#<id>] or 'closes [#<id>]' for each (ADR-65/66 forward-only index).
```

**N3b — genuine stale canonical doc (BLOCKED; HEAD unmoved):**
```
canonical_freshness last_reviewed gate (A2 FAIL blocks the commit).......................Failed
- hook id: canonical_freshness
- exit code: 1

canonical_freshness FAIL: ARCHITECTURE.md: last_reviewed 2026-07-01 predates last edit
2026-07-08 - edited but not re-reviewed
canonical_freshness: 1 canonical doc(s) stale (edited since review) — re-read end-to-end
and bump last_reviewed to the GENUINE review date (never a fake stamp). Bypass in good
faith with --no-verify if wrong.
```

**N4 — direct commit on `main` (NOT blocked — the finding):**
```
canonical_freshness last_reviewed gate (A2 FAIL blocks the commit).......................Passed
Require [#id] in commit message when a BACKLOG task is closed.......................Passed
[main 6a44191] test(qa-n4): direct commit on main — probe branch protection
 1 file changed, 1 insertion(+)
 create mode 100644 qa-n4-probe.txt
```
(Reverted immediately: `main` = `origin/main` = `fba7b13`, no push.)

**Action 3 — SessionStart:resume legs:**
```
leg 1  check_floor_hash.py --require-present      -> exit 0
leg 2  python -m pre_commit install  (.venv)      -> exit 1
       .venv\Scripts\python.exe: No module named pre_commit
WMI    platform.machine() under Python312         -> AMD64  took=0.482 s
```

## Findings

- **F1 — no consumer-side commit-time branch protection on `main` (N4).** A direct
  commit to `main` lands with nothing blocking it: the content gates are
  *content-scoped* and no deployed hook inspects the current branch. The
  branch→merge-`--no-ff` invariant (core-invariant #5 / floor Ship-rule) currently
  rests on (a) operator discipline, (b) the session-end `session_end_backpressure`
  Stop gate (ADR-85 — a *session-end* block, not a *commit-time* one), and (c) any
  server-side rule. **Route to hub:** does the hub carry a local block-direct-to-main
  hook that consumers lack? *Severity: medium.* (Measured, not fixed — C-S3v2.)
- **F2 — SessionStart:resume auto-arm still fails in-venv (G8; Action 3).** The active
  `.venv` lacks `pre_commit` despite the `pre-commit>=4.5` `[dev]` declaration
  (`pyproject.toml`, `d6dc783`) — `pip install -e ".[dev]"` has not been re-run into
  this venv. The resume-arm is a fast non-blocking fail; **commit-time enforcement is
  unaffected** (git shims hardcode `INSTALL_PYTHON`=Python312, which has `pre_commit`).
  *Severity: low* — self-heals on the next `[dev]` install.
- **F3 — pre-push stage armed-but-empty (A1).** `.git/hooks/pre-push` is installed and
  invokes `pre_commit hook-impl --hook-type=pre-push`, but the deployed config declares
  no pre-push-stage hooks (`default_stages: [pre-commit]`); the stage runs zero checks
  and the push proceeds. *Severity: informational* — by-design for this config; matters
  only if a push-time gate is ever expected.
- **Nuance — `canonical_freshness` A2 is date-granular / same-day-safe (N3a).** Not a
  defect (documented design): a canonical doc edited *and* reviewed on the same calendar
  day does not trip A2. Teeth require edit-date > `last_reviewed` by ≥1 day (N3b proved
  the teeth). Worth knowing when reasoning about same-day doc work.
- **Correction — WMI hang (G9) currently dormant.** The 2026-07-08 journal's
  "BLOCKED-by-WMI" status did **not** reproduce this session; `platform.machine()` was
  responsive (0.482 s) and all 6 commit-time invocations completed <2 s. G9 remains a
  latent, intermittent hazard on the Python312 path — not an active blocker today.

## Verdict

**The deployed methodology enforces in effect — with the exceptions below.** Every
commit-time gate probed adversarially blocked the violation it targets, each with the
commit refused and `HEAD` unmoved and a verbatim, actionable refusal: `floor-hash-verify`
(N1), `backlog-id-on-close` at the commit-msg stage (N2), and `canonical_freshness` A2
on a genuinely stale doc (N3b). The positive path passes honest work cleanly (A1; suite
432 green), and the skills/plugin leg resolves and runs (`/codex-review` end-to-end;
`/review-closures` surfacing, approving nothing). **Exceptions:** (F1) there is **no
consumer-side commit-time block on direct-to-`main` commits** — the branch→merge
invariant is enforced by discipline + the session-end Stop gate, not a hook (routed to
the hub); (F2) the SessionStart:resume auto-arm still fails in-venv on module resolution
(G8), though commit-time enforcement is intact; (F3) the pre-push stage is armed but
empty; and one by-design granularity nuance in `canonical_freshness`. The WMI hang (G9)
did not reproduce.

## Residue statement

Zero residue. All negative-probe injections reverted (verified `git diff` vs `main`
empty on every non-report path); the direct-to-`main` probe commit hard-reset away
(`main` = `origin/main` = `fba7b13`, never pushed); the `/codex-review` artifact
removed; the scratch branch `qa/lived-exercise-2026-07-09` (local + `origin`) deleted at
close-out. Only this report + its JOURNAL entry land on `main`, via `docs/qa-lived-exercise`
→ `--no-ff`.
