# Violation-proof transcripts — #67 sealed-key guard, #68 docs/ registry guard

**Date:** 2026-07-19
**Branch:** `feat/c-guards`
**Guards under proof:** `scripts/validate_sealed_keys.py` (#67), `scripts/validate_docs_registry.py` (#68)
**Method:** every claim below was produced by *attempting the violation*. The existence of a hook file is not evidence. Where a real `git commit` was attempted, HEAD is shown before and after.

All fixtures were dummy files. **No real sealed key was created, opened, moved, or staged**, and `docs/audits/2026-07-18-cli4-parity/` was not touched.

---

## Summary

| # | Case | Expected | Result |
|---|---|---|---|
| 1a | Stage `SEALED-KEY.json` | reject | reject, HEAD unchanged |
| 1b | Stage `...-KEY-SEALED.json` (epi1 shape) | reject | reject, exit 1 |
| 2a | Stage `docs/smoke/report.md` | reject | reject, exit 1 |
| 2b | Stage `docs/decisions/archive/` (taxonomy class b) | pass | pass, exit 0 |
| 2c | Unregistered `docs/audits/<corpus>/` | reject | reject, exit 1 |
| 2d | **Same** dir after adding its registry row | pass | pass, exit 0 |
| 2e | **Same** dir after restoring the registry | reject | reject, exit 1 |
| 3a | `Live corpora` heading renamed | GUARD MALFUNCTION | malfunction, exit 1 |
| 3b | Registry file absent | GUARD MALFUNCTION | malfunction, exit 1 |
| 3c | Registry table reformatted | GUARD MALFUNCTION | malfunction, exit 1 |
| 4a | Override naming a *different* path | still reject | reject, exit 1 |
| 4b | Override naming the *exact* path | bypass + banner | bypass, exit 0, banner emitted |
| 5 | Sealed-key guard on a git error | fail CLOSED | malfunction, exit 1 |
| 6 | Pattern vs `git ls-files` repo-wide | zero matches | zero matches |
| 7 | Both guards on the clean tree | pass | pass |

---

## 1 — #67 rejects a staged sealed key

### 1a — literal `SEALED-KEY.json`, via a real `git commit`

```
$ git add SEALED-KEY.json
$ git diff --cached --name-only
SEALED-KEY.json
$ git commit -m "test: deliberate violation - must be rejected"

Reject a staged sealed key (#67)......................................Failed
- hook id: validate-sealed-keys
- exit code: 1

validate_sealed_keys: refused -- sealed key staged (#67):
  SEALED-KEY.json
  A sealed key must never enter the repo: it holds the blind-trial identity mapping, and committing it destroys the seal.
  These files are meant to stay gitignored and untracked -- if one is staged, an ignore rule was broken or bypassed (e.g. by a `git mv`).
  Unstage it (`git restore --staged <path>`) and check `.gitignore`. If committing it is genuinely intended, authorize the exact path: AICOUNCIL_SEALED_KEY_ALLOW='<repo-relative-path>' (`;`-separated for several). Do NOT use --no-verify -- it disarms every other hook too.

HEAD before: 11693f9    HEAD after: 11693f9   (unchanged — the commit did not happen)
```

### 1b — the epi1 name shape, which #67's literal spec would have missed

This is the case that justified widening the pattern. The two real keys are named inconsistently — `SEALED-KEY.json` and `2026-07-17-epi1-archaeology-KEY-SEALED.json` — so a literal `SEALED-KEY*.json` guard would have covered only one of them.

```
$ git add 2026-07-17-epi1-archaeology-KEY-SEALED.json
$ git commit -m "test: deliberate violation - epi1 name shape"

validate_sealed_keys: refused -- sealed key staged (#67):
  2026-07-17-epi1-archaeology-KEY-SEALED.json
  [...same guidance...]

git commit exit code: 1
HEAD: 11693f9 (unchanged)
```

---

## 2 — #68 rejects an unregistered directory, and passes the registered ones

### 2a — `docs/smoke/`, recreating the real 2026-07-18 incident

```
$ git add docs/smoke/report.md
$ git commit -m "test: deliberate violation - unregistered docs dir"

validate_docs_registry: refused -- unregistered new docs/ directory (#68):
  'docs/smoke/' is a new directory under docs/ that is neither a sanctioned taxonomy folder nor a registered live corpus (registries live in docs/audits/README.md).
  To register a live corpus: add an essence markdown at the parent root AND a row to the 'Live corpora' table in docs/audits/README.md naming the path, what it is, the ruling that keeps it there, its essence markdown, and its exit condition. Otherwise the artifact belongs in an existing folder.

git commit exit code: 1
HEAD: 11693f9 (unchanged)
```

### 2b — a sanctioned `archive/` child passes (proving this is not a blanket ban)

```
$ git add docs/decisions/archive/note.md
$ python scripts/validate_docs_registry.py
   exit=0 (pass)
```

### 2c–2e — the decisive test: one directory, three registry states

The same path is used throughout. Only `docs/audits/README.md` changes. This proves the verdict is driven by the registry *read at runtime*, not by anything hardcoded in the script.

```
2c) registry UNCHANGED — docs/audits/2026-07-20-testcorpus/ staged
    validate_docs_registry: refused -- unregistered new docs/ directory (#68):
      'docs/audits/2026-07-20-testcorpus/' is a new directory in docs/audits/ with no row
      in the 'Live corpora' table of docs/audits/README.md. An unregistered folder is
      indistinguishable from a leftover.
    exit=1 (block)

2d) one row added to the Live-corpora table for that exact path — nothing else changed
    exit=0 (pass)

2e) registry restored (verified byte-identical to HEAD: `git diff --quiet` clean)
    same block message as 2c
    exit=1 (block)
```

Registered live corpora as parsed from the live README at proof time:

```
registered corpora : ['2026-07-17-epi1-archaeology', '2026-07-18-cli4-parity']
taxonomy dirs      : ['archive']
```

Both currently-registered corpora and `docs/audits/archive/` are admissible; `docs/audits/archive/legacy/` is grandfathered (tracked in HEAD) and confirmed not to fire.

---

## 3 — #68 fails CLOSED, and says so

Required because a registry guard that fails open silently passes everything — the same false confidence that made a narrow #67 pattern unacceptable. A malfunction must be distinguishable from a policy violation at a glance, or the first false block trains everyone to bypass the gate.

The registry is parsed on **every** invocation, not only when a `docs/` directory is staged, so a broken registry surfaces immediately rather than lying dormant until the next corpus is added.

```
3a) 'Live corpora' heading renamed
    validate_docs_registry: GUARD MALFUNCTION -- this is NOT a policy violation.
      Registry: docs/audits/README.md
      Problem:  no 'Live corpora' section found (heading renamed, moved, or removed)
      Nothing is wrong with what you staged. The guard cannot read its own registry, so it
      cannot tell a registered folder from a leftover.
    exit=1

3b) registry unreadable from the index (staged for deletion, or replaced by a non-file entry)
      Problem:  could not read docs/audits/README.md from the git index -- it may be staged
                for deletion or replaced by a non-file entry
                (fatal: path 'docs/audits/README.md' exists on disk, but not in the index)
    exit=1

3c) table reformatted (Path column de-backticked)
      Problem:  the 'Live corpora' table has rows but no parseable path in its first column
                (expected a backticked `<dir>/`; column order or formatting changed)
    exit=1
```

`docs/audits/README.md` was restored after each sub-case and verified byte-identical to HEAD.

> **Transcript currency.** Cases 3a–3c were first recorded against the pre-hardening
> implementation; the sol/terra fix rounds changed 3b's message (the registry is now read from
> the **index**, not the working tree). All three were **re-run against the final code** and the
> text above is the final output.

### 3d — each registered corpus proven by an actual staged attempt

Closing an evidence gap terra identified: the earlier draft asserted that both live corpora
pass, but had only staged one of them.

```
P1) docs/audits/2026-07-18-cli4-parity/new-section/f.md   staged   -> exit=0 (pass)
P2) docs/audits/2026-07-17-epi1-archaeology/f.md          staged   -> exit=0 (pass)
P3) docs/audits/archive/probe.md                          staged   -> exit=0 (pass)
```

P2 required `git add -f` because that corpus is gitignored (`.gitignore:60`) and therefore
absent from every worktree — recorded rather than glossed, since it is the same
worktree/primary asymmetry that drove the staged-only design (R2) and finding F-B.

---

## 4 — The #67 override, and its audit-trail limitation

**The override exists and is documented here because it leaves no other trace.**

- **Variable:** `AICOUNCIL_SEALED_KEY_ALLOW`
- **Invocation:** `AICOUNCIL_SEALED_KEY_ALLOW='<repo-relative-path>' git commit ...`, `;`-separated for several paths.
- **Constraint:** it must name the **exact repo-relative path**. A bare truthy value authorizes nothing, so the override can never blanket-disarm the guard.
- **Why not `--no-verify`:** that disarms every other hook at the same time. This override is scoped to this one gate and leaves the rest armed.
- **Audit trail — the limitation:** an environment variable is invisible in `git log`. A bypass of the sealed-key guard **cannot be seen after the fact from the repository alone.** The stderr banner below is its only record, and it lives only in the terminal transcript of whoever ran the commit.

```
4a) override naming a DIFFERENT path than the staged one -- must still block
    validate_sealed_keys: refused -- sealed key staged (#67):
      2026-07-17-epi1-archaeology-KEY-SEALED.json
    exit=1 (block)

4b) override naming the EXACT staged path
    ========================================================================
    validate_sealed_keys: SEALED-KEY GUARD DELIBERATELY BYPASSED
      ALLOWED: 2026-07-17-epi1-archaeology-KEY-SEALED.json
      Authorized by the AICOUNCIL_SEALED_KEY_ALLOW environment variable.
      A sealed key is being committed on purpose. This bypass leaves NO record in git log
      -- this terminal transcript is its only audit trail.
    ========================================================================
    exit=0 (bypass)
```

---

## 5 — #67 also fails CLOSED

The peer casing gate (`validate_audit_casing.py`) fails *open* because it polices a naming convention. This one polices a secret, so the failure direction is inverted.

```
$ cd /c && python .../scripts/validate_sealed_keys.py
validate_sealed_keys: GUARD MALFUNCTION -- this is NOT a policy violation.
  Could not read the staged file list: error: unknown option `cached'
  Blocking the commit because this gate protects a secret and cannot verify that no sealed key is staged.
  Fix the git error and retry; do not bypass.
exit=1
```

---

## 6 — The #67 pattern matches zero tracked files

A guard that fires on an already-tracked file blocks every future commit. Re-run at proof time:

```
$ git ls-files | grep -iE '(^|/)[^/]*(sealed[^/]*key|key[^/]*sealed)[^/]*\.json$'
   (zero matches)
$ git ls-files '*.json'
.claude/settings.json
.vscode/settings.json
```

Two tracked `.json` files repo-wide; neither matches.

---

## 7 — Neither guard fires on the clean tree

```
$ python -m pre_commit run validate-sealed-keys --all-files
Reject a staged sealed key (#67).........................................Passed
$ python -m pre_commit run validate-docs-registry --all-files
Registry check for new docs/ directories (#68)...........................Passed
```

All fixtures were removed; `git status --short` is empty. The only ignored artifact left behind is `.ruff_cache/`, which is gitignored at `.gitignore:7` and regenerated by every ruff run.

### 7b — full pre-merge gate (criterion 3)

Run from the worktree with `PYTHONPATH` set to the worktree's `src/` — a bare `check.ps1` from a
worktree tests the *primary* checkout's code via the shared editable install.

```
$ $env:PYTHONPATH = "<worktree>\src"; .\scripts\check.ps1
Running pytest...   543 passed, 6 deselected in 69.60s
Running mypy...     Success: no issues found in 38 source files
Running ruff...     All checks passed!
=== check.ps1 exit: 0 ===
```

No existing hook was removed or modified — the two new stanzas are additive; `git diff main...HEAD`
on `.pre-commit-config.yaml` shows insertions only.

---

## What is NOT proven — known limits

Recorded so the next reader does not over-trust these guards.

1. **#68 polices what git can see.** An empty directory can never enter the repo and is therefore out of scope by ruling, not by oversight (see #68's done-when). A populated unauthorized directory *is* caught; empty residue on local disk is not, and belongs to the `check.ps1` / `verify_*` family.
2. **#68's grandfathering is by design.** Every directory already tracked in HEAD is admissible without a registry row. Existing unregistered directories are therefore not retroactively policed — the guard is prospective-only, matching the peer casing gate.
3. **Anything inside an `archive/` is deliberately out of scope.** Invariant class b reads *"`archive/` — governed by its own `archive/README.md`"*, so the Live-corpora table does not police archive contents. A new directory at `docs/audits/archive/<anything>/` therefore passes. This is a deliberate scoping decision, not an oversight — see the finding below.
4. **The registry parse is shape-dependent.** It requires a backticked `` `<dir>/` `` in the first column of the Live-corpora table. That is deliberately strict and fails closed (case 3c) rather than degrading — but it does mean a well-intentioned reformat of the README will block commits until fixed.
5. **The #67 override has no repository-side audit trail** (§4 above). If that becomes unacceptable, the override needs to move to something `git log` can see.
6. **Neither guard was exercised on a merge commit or a rebase**, only on ordinary `git commit`.
7. **Registration is checked as "has a registry row", not "has a row AND an essence markdown".**
   The README's invariant defines it as both. The essence column cannot be mechanically checked
   today because it legitimately holds prose for a not-yet-written artifact — the `cli4-parity`
   row's essence cell reads *"The parity report written at unseal"*. So the #68 claim is
   narrower than the written invariant: it enforces the row, not the essence file. (terra finding.)
8. ~~No unit tests.~~ **RESOLVED.** `tests/test_validate_sealed_keys.py` and
   `tests/test_validate_docs_registry.py` added on operator instruction, following the
   `tests/test_validate_audit_casing.py` precedent (direct classifier tests + end-to-end against
   a real temp git repo). **43 tests**, one named regression per bypass found by sol, terra, or
   self-review.

   The regression tests were themselves verified: both guards were temporarily reverted to the
   pre-fix `--name-only` (no `-z`) form and the two unicode tests **failed** as they must, then
   passed again once restored. A regression test that passes against the broken code is
   worthless, so this check is the point.

   ```
   pre-fix guards:  2 failed  (unicode dir BYPASSED the guard: assert 0 == 1)
   fixed guards:    43 passed
   ```
9. **A flat or packed corpus at the audits root** (`raw-trial-001.json …`, or a `.zip`) creates
   no directory and is not caught. That is invariant class (a) enforcement — a different guard
   from #68's directory registry check.
10. ~~The #67 guard was not re-tested against the Unicode/quoting path.~~ **Tested — it was
    bypassed too, and is now fixed.** Writing this up as an unverified assumption prompted the
    test, which failed:

    ```
    $ git add -f "docs/évasion/SEALED-KEY.json"
    $ git diff --cached --diff-filter=A --no-renames --name-only
    "docs/\303\251vasion/SEALED-KEY.json"
    $ python scripts/validate_sealed_keys.py
       exit=0            <-- SEALED KEY ADMITTED
    ```

    The C-quoted path carries a **trailing** `"`, which defeats the `\.json$` anchor. Fixed by
    the same `-z` change applied to #68. Re-proven: the Unicode key now blocks (exit 1), the
    ASCII key still blocks, and the clean tree still passes.

---

## Findings raised during proof

### F-A — #68 initially blocked the sanctioned corpus-exit path (FIXED)

Testing #27's required unseal move revealed a false positive: a corpus moving to
`docs/audits/archive/<corpus>/` was **rejected**, because its parent is `docs/audits/archive`
rather than `docs/audits` and its name is not itself `archive`.

```
before fix:  'docs/audits/archive/2026-07-18-cli4-parity/' is a new directory under docs/
             that is neither a sanctioned taxonomy folder nor a registered live corpus
             exit=1  (blocks the exact move #27 must perform at unseal)

after fix:   exit=0
```

Fix: anything **inside** an admissible taxonomy folder is governed by that folder's own
README (invariant class b), not by the Live-corpora table. Regression-checked — every case in
§2 still behaves as proven, and `docs/smoke/`, `docs/decisions/sneaky/`, and an unregistered
`docs/audits/<corpus>/` all still block.

Consequence, stated plainly: `docs/audits/archive/` contents are **not** policed by this guard.
The archive's own README governs them.

### F-C — sol adversarial pass found four real bypasses of #68 (ALL FIXED)

The adversarial pass was run precisely because *a registry guard that can be trivially bypassed
is worse than none.* It found four. All were reproduced before fixing and re-proven after.

| # | Bypass | Before | After |
|---|---|---|---|
| 1 | **Non-ASCII directory name.** `core.quotePath` defaults on, so `--name-only` C-quotes `docs/audits/évasion/f.md` as `"docs/audits/\303\251vasion/f.md"`. The leading `"` makes `parts[0] != "docs"`, so the path is skipped — **the guard was fully bypassed**. | exit 0 | exit 1 |
| 2 | **`docs/` token injected into the invariant table.** Any backticked `` `docs/` `` anywhere in that table became an admissible taxonomy name matched at *any* depth, admitting everything. | exit 0 | exit 1 |
| 3 | **Staged registry deletion masked by an untracked copy.** `git rm --cached docs/audits/README.md` stages the deletion while leaving the file on disk; the guard parsed the working-tree copy and passed. | exit 0 | exit 1 (malfunction) |
| 4 | **Gitlink / directory symlink.** `git submodule add <url> docs/rogue` stages one path with no child component, so no directory prefix was derived. | not detected | treated as a directory |

Fixes: read staged paths NUL-delimited via `--raw -z` (kills 1, and yields the mode for 4);
reserve `docs`/`.`/`..` as never-admissible taxonomy names and require single-segment tokens
(kills 2); read the registry from the **index** via `git show :<path>` rather than the working
tree (kills 3).

A fifth hardening was applied alongside: a taxonomy folder is now admissible only at a
sanctioned depth (`docs/<taxonomy>` or `docs/<section>/<taxonomy>`). Previously `archive` matched
at any depth, so `docs/a/b/archive/rogue/` would have laundered anything beneath it.

Re-proof after fixing — every earlier case still behaves, and the deep-laundering shape blocks:

```
  ok  BLOCK  docs/a/b/archive                          ok  PASS   docs/audits/archive
  ok  BLOCK  docs/a/b/archive/rogue                    ok  PASS   docs/decisions/archive
  ok  BLOCK  docs/smoke                                ok  PASS   docs/audits/archive/legacy
  ok  BLOCK  docs/audits/2026-07-20-unreg              ok  PASS   docs/audits/archive/2026-07-18-cli4-parity
                                                       ok  PASS   docs/audits/2026-07-18-cli4-parity
```

**Two sol findings were deliberately NOT fixed**, and are recorded as accepted scope limits:

- *A flat or packed corpus at the audits root* (`docs/audits/raw-trial-001.json …`, or a `.zip`)
  creates no new directory and is not caught. Enforcing invariant class (a) — root entries must be
  date-slug markdown — is a **different** guard from #68's directory registry check, and adding it
  would exceed this lane's frozen contract. Worth a future ticket; not filed here.
- *Removing a row while its directory stays tracked* is the stale-row case already recorded as
  **F-B** below, deliberately not mechanised.

### F-B — #68 does not detect a stale registry row

#68 catches *a corpus with no row*. It does **not** catch *a row with no corpus* — a registry
row left behind after its corpus has exited to `archive/`. Both are "table rot"; only the first
is mechanised.

This matters for #27, whose done-when carries the by-hand obligation to retire the
`2026-07-18-cli4-parity` row at unseal *"so the table cannot rot while #68 is unbuilt."* #68
being armed does not fully replace that obligation.

A disk-based stale-row check is also not straightforwardly available: the `2026-07-17-epi1-archaeology`
row is gitignored, so its directory is legitimately absent in a worktree and present in the
primary checkout — the exact worktree/primary divergence the staged-only design (R2) was chosen
to avoid.

Raised to the operator rather than resolved unilaterally.
