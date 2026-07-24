# Codex Review — validate-claims-unit1

**Date:** 2026-07-24
**Branch:** `feat/validate-claims-unit1`
**HEAD:** `66bb5b1`
**Diff range:** `main..feat/validate-claims-unit1`
**Codex version:** codex-cli 0.145.0
**Mode:** diff-review

---

## Focus

- Finding model: is the shlex round-trip + bare-arg0 contract sound? any way prose evidence or a non-runnable command slips through?
- Exit-code contract: does a crashing leg truly force >=2, and is the check.ps1 swallow (explicit exit 0) correct so findings never fail the gate but a crash surfaces Red?
- R2 precision: are the guards/allowlist principled, or do they suppress true drift / admit false positives?
- R3/R4 mention-based checks: any false-negative (a real roster gap that mention-matching hides)?
- R8 reachability: is rev-list --all + cat-file --batch-check correct for 'reachable from any ref', and free of injection via doc-sourced SHA tokens?
- Read-only guarantee: any code path that could mutate the repo?

---

## Findings
## Critical

(none)

## High

### `scripts/validate_claims.py:401`

**What:** Failed `git rev-list`/`cat-file` subprocesses are converted into empty results rather than checker errors.  
**Why:** A broken Git query can yield a clean/pass report and exit 0, so the advertised `>=2` error contract is bypassed; `check.ps1` will not show Red.  
**Fix direction:** Treat non-zero Git query results as leg errors so `run_all()` produces exit 2; retain exit 0 swallowing only for genuine findings.

### `scripts/validate_claims.py:76`

**What:** `Finding` accepts any argv after a permitted runner, including non-runnable evidence such as `("python",)` or `("git", "not-a-command")`.  
**Why:** Shlex round-tripping proves quoting only; it does not prove the reported evidence is a usable command, despite the rule-12 contract.  
**Fix direction:** Validate supported runner-specific command shapes and add negative tests for bare/invalid runner invocations.

### `scripts/validate_claims.py:198`

**What:** R2 ignores root-level file references and any typo whose first path segment does not already exist, e.g. `` `CONTRIBUTNG.md` `` or `` `scrips/check.ps1` ``.  
**Why:** These are exactly stale local-path claims R2 should catch, but the slash and existing-top-level-directory guards silently suppress them.  
**Fix direction:** Use explicit external-path exceptions/allowlists rather than treating an absent top-level segment as non-local.

### `scripts/validate_claims.py:310`

**What:** R3 and R4 treat any whole-token mention in the enclosing section as roster membership (`:364`), while their evidence scans the entire document (`:319`, `:371`).  
**Why:** A hook/ADR mentioned in a caveat or prose can hide a missing roster entry; conversely, emitted evidence can show no gap when the identifier appears outside the roster.  
**Fix direction:** Parse the actual roster entries structurally and use that same extracted roster for both detection and evidence.

### `scripts/validate_claims.py:448`

**What:** R8 evidence checks ancestry only against `HEAD`, not reachability from every ref.  
**Why:** A commit retained exclusively by another branch is not an ancestor of `HEAD`; the evidence therefore cannot establish the finding’s claimed “unreachable from any ref” reality.  
**Fix direction:** Emit evidence that tests all refs using the same any-ref reachability criterion as the checker.

## Medium

(not assessed — default diff review is restricted to Critical/High.)

## Low

(not assessed — default diff review is restricted to Critical/High.)

No repo-mutating path was found in the newly added checker.
