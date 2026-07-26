# Codex Review — 116-resolution-model

**Date:** 2026-07-26
**Branch:** `fix/111-token-log-address`
**HEAD:** `a85de04`
**Diff range:** `main..fix/111-token-log-address`
**Codex version:** codex-cli 0.145.0
**Mode:** diff-review

---

## Focus

- Can the R2 resolution model silently suppress a real missing path via the declared bases or the two exclusion lists?
- Is _R2_RUNTIME_PATHS a principled declaration or a disguised allowlist?
- Does resolving against the tracked tree actually make the verdict commit-deterministic, or is there a residual disk dependency?
- Is the ./ normalization correct, and can it mask a distinct path?
- Do the self-validating tests for bases and runtime paths actually fail if the declaration is wrong?

---

## Findings
## Critical

(none)

## High

### scripts/validate_claims.py:261 — “Tracked tree” is read from the mutable index, not a commit

**What:** `git ls-files` reads the current index; staged additions/removals can change R2’s verdict for the same `HEAD` commit.  
**Why:** This retains checkout-state dependence and can silently pass a claim solely because a path is staged but not committed; the evidence command has the same issue.  
**Fix direction:** Resolve both scanned documents and paths from one explicit Git tree object (for example, `HEAD` or a supplied revision), and add a test that stages a path without committing it.

### scripts/validate_claims.py:275 — Any-base fallback can mask a missing path at its written location

**What:** A token is accepted when it exists under any declared base, so a root-relative missing `research/merger.py` is silently accepted because `src/ai_council/research/merger.py` exists.  
**Why:** The token has no source-base context, so unrelated base expansions turn an ambiguous reference into a false pass. The current tests cover intended base-relative successes but not this collision.  
**Fix direction:** Bind bases to document/section context or require explicit base-qualified paths; add a collision test where only an unintended expansion exists.

### scripts/validate_claims.py:401 — Runtime-prefix exclusion is a broad allowlist without per-claim validation

**What:** Every descendant of `output/` or `council_inbox/archive/` is skipped, including a nonexistent non-runtime path such as `output/contract.md`.  
**Why:** `_R2_RUNTIME_PATHS` is behaviorally an allowlist, and its self-test only proves each root is currently ignored/untracked—not that every suppressed descendant is a valid runtime artifact. It also uses `git check-ignore`, which can be satisfied by untracked local/global ignore configuration.  
**Fix direction:** Limit exclusions to documented runtime artifact patterns or contextual claims, and validate against tracked `.gitignore` rules rather than local Git ignore configuration.

## Medium

(none)

## Low

(none)
